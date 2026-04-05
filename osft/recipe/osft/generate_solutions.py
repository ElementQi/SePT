# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Generate solutions from a dataset at specified temperatures and save as parquet files.

Each temperature produces a separate parquet file containing the original dataset
columns plus a `solution` column with the generated text.

Reuses generation infrastructure from tau_sweep.py.
"""

import os
from pathlib import Path
from pprint import pprint

import hydra
import pandas as pd
import ray

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

from omegaconf import OmegaConf
from torchdata.stateful_dataloader import StatefulDataLoader

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.utils import hf_tokenizer
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.fs import copy_to_local
from verl.workers.fsdp_workers import ActorRolloutRefWorker

from recipe.osft.tau_sweep import (
    _load_model_path_from_hydra,
    _load_tokenizer_with_fallback,
    _resolve_tau_values,
    create_rl_dataset,
)


@hydra.main(config_path="config", config_name="generate_solutions", version_base=None)
def main(config):
    run_generation(config)


def run_generation(config) -> None:
    if not ray.is_initialized():
        ray.init(
            runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}},
            num_cpus=config.ray_init.num_cpus,
        )

    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)
class TaskRunner:
    def run(self, config):
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        trust_remote_code = config.data.get("trust_remote_code", False)
        ckpt_path = config.eval.get("ckpt_path", None)

        if ckpt_path:
            ckpt_path = str(Path(ckpt_path).expanduser())
            tokenizer = _load_tokenizer_with_fallback(config, run_dir=None, first_ckpt_path=ckpt_path)
        else:
            local_path = copy_to_local(config.model.path, use_shm=config.model.get("use_shm", False))
            tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)

        ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(ActorRolloutRefWorker), config=config, role="rollout")
        resource_pool = RayResourcePool(
            process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes,
            use_gpu=True,
            max_colocate_count=1,
        )
        wg = RayWorkerGroup(
            resource_pool=resource_pool,
            ray_cls_with_init=ray_cls_with_init,
            device_name=config.trainer.device,
        )
        wg.init_model()

        if ckpt_path:
            wg.load_checkpoint(local_path=ckpt_path, hdfs_path=None, del_local_after_load=False)

        data_path = config.data.get("val_files", config.data.path)
        dataset = create_rl_dataset(data_path, config.data, tokenizer, None)

        batch_size_cfg = config.data.get("batch_size", None)
        if batch_size_cfg is None or int(batch_size_cfg) <= 0:
            batch_size = len(dataset)
        else:
            batch_size = min(int(batch_size_cfg), len(dataset))

        dataloader = StatefulDataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=config.data.get("dataloader_num_workers", 8),
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
        )
        assert len(dataloader) >= 1, "Dataloader is empty!"

        output_dir = config.data.get("output_dir", "./generated_solutions")
        os.makedirs(output_dir, exist_ok=True)

        tau_values = _resolve_tau_values(config.eval)
        for tau in tau_values:
            do_sample = tau > 0.0
            gen_kwargs = OmegaConf.create(
                {
                    "temperature": tau,
                    "top_p": config.rollout.top_p,
                    "top_k": config.rollout.top_k,
                    "do_sample": do_sample,
                }
            )
            pprint(f"Generating solutions at temperature={tau}")
            rows = _generate(config, tokenizer, dataloader, wg, gen_kwargs)

            tau_str = f"{tau}".replace(".", "s")  # e.g. 0.6 -> 0s6
            output_path = os.path.join(output_dir, f"solutions_tau{tau_str}.parquet")
            df = pd.DataFrame(rows)
            df.to_parquet(output_path, index=False)
            pprint(f"Saved {len(df)} solutions (tau={tau}) to {output_path}")


def _generate(config, tokenizer, dataloader, wg, gen_kwargs):
    """Generate solutions for every sample in the dataloader at the given temperature.

    Returns a list of dicts, each containing the original metadata fields
    (id, data_source, ability, reward_model) and a ``solution`` field with
    the generated text.
    """
    rows = []

    for batch_data in dataloader:
        batch = DataProto.from_single_dict(batch_data)

        # Optionally repeat samples (e.g. to draw multiple solutions per prompt)
        n_samples = config.data.get("n_samples", 1) if gen_kwargs.do_sample else 1
        if n_samples > 1:
            batch = batch.repeat(repeat_times=n_samples, interleave=True)

        # Snapshot the metadata we want to preserve before popping tensors
        ntb = batch.non_tensor_batch
        ids = ntb.get("id", [""] * len(batch))
        data_sources = ntb.get("data_source", ["unknown"] * len(batch))
        abilities = ntb.get("ability", [None] * len(batch))
        reward_models = ntb.get("reward_model", [None] * len(batch))

        # Build the generation-only batch (mirrors tau_sweep._validate)
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
        if "multi_modal_data" in ntb:
            non_tensor_batch_keys_to_pop.append("multi_modal_data")
        if "raw_prompt" in ntb:
            non_tensor_batch_keys_to_pop.append("raw_prompt")
        if "tools_kwargs" in ntb:
            non_tensor_batch_keys_to_pop.append("tools_kwargs")

        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
        )

        gen_batch.meta_info = {
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "recompute_log_prob": False,
            "do_sample": gen_kwargs.do_sample,
            "temperature": gen_kwargs.temperature,
            "top_p": gen_kwargs.top_p,
            "top_k": gen_kwargs.top_k,
            "validate": True,
        }

        gen_batch_padded, pad_size = pad_dataproto_to_divisor(gen_batch, wg.world_size)
        output_padded = wg.generate_sequences(gen_batch_padded)
        output = unpad_dataproto(output_padded, pad_size=pad_size)

        response_ids = output.batch["responses"]
        solutions = [tokenizer.decode(token_ids, skip_special_tokens=True) for token_ids in response_ids]

        for i, solution in enumerate(solutions):
            rows.append(
                {
                    "id": ids[i],
                    "data_source": data_sources[i],
                    "ability": abilities[i],
                    "reward_model": reward_models[i],
                    "solution": solution,
                }
            )

    return rows


if __name__ == "__main__":
    main()
