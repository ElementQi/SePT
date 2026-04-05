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
Generate responses given a dataset of prompts
"""

import os
from pathlib import Path

import hydra
import numpy as np
import ray
import yaml

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import json
from collections import defaultdict, deque
from pprint import pprint

from omegaconf import OmegaConf
from torchdata.stateful_dataloader import StatefulDataLoader

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.trainer.ppo.metric_utils import (
    process_benchmark_metrics,
)
from verl.trainer.ppo.ray_trainer import compute_response_mask
from verl.trainer.ppo.reward import compute_reward_async, load_reward_manager
from verl.utils import hf_tokenizer
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.fs import copy_to_local
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.fsdp_workers import ActorRolloutRefWorker


@hydra.main(config_path="config", config_name="sept_generation", version_base=None)
def main(config):
    run_generation(config)


def run_generation(config) -> None:
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(
            runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}},
            num_cpus=config.ray_init.num_cpus,
        )

    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))
@ray.remote(num_cpus=1)
class TaskRunner:
    def run(self, config):
        pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
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
        resource_pool = RayResourcePool(process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes, use_gpu=True, max_colocate_count=1)

        wg = RayWorkerGroup(
            resource_pool=resource_pool,
            ray_cls_with_init=ray_cls_with_init,
            device_name=config.trainer.device,
        )
        wg.init_model()

        if ckpt_path:
            wg.load_checkpoint(local_path=ckpt_path, hdfs_path=None, del_local_after_load=False)

        val_dataset = create_rl_dataset(config.data.get("val_files", config.data.path), config.data, tokenizer, None)
        # Avoid one-giant-batch validation: reduces SPMD stragglers and improves GPU utilization.
        val_bs_cfg = config.data.get("batch_size", None)
        if val_bs_cfg is None or int(val_bs_cfg) <= 0:
            val_batch_size = len(val_dataset)
        else:
            val_batch_size = min(int(val_bs_cfg), len(val_dataset))
        val_reward_fn = load_reward_manager(config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {}))
        val_dataloader = StatefulDataLoader(
            dataset=val_dataset,
            batch_size=val_batch_size,
            num_workers=config.data.get("dataloader_num_workers", 8),
            shuffle=config.data.get("validation_shuffle", False),
            drop_last=False,
            collate_fn=collate_fn,
        )
        assert len(val_dataloader) >= 1, "Validation dataloader is empty!"

        from verl.utils.tracking import Tracking

        validation_generations_logger = ValidationGenerationsLogger()
        logger = Tracking(
            project_name=config.trainer.project_name,
            experiment_name=config.trainer.experiment_name,
            default_backend=config.trainer.logger,
            config=OmegaConf.to_container(config, resolve=True),
        )

        tau_values = _resolve_tau_values(config.eval)
        for global_step, tau_eval in enumerate(tau_values):
            do_sample = tau_eval > 0.0
            val_gen_kwargs = OmegaConf.create(
                {
                    "temperature": tau_eval,
                    "top_p": config.rollout.top_p,
                    "top_k": config.rollout.top_k,
                    "do_sample": do_sample,
                }
            )
            val_metrics = _validate(config, tokenizer, val_dataloader, val_reward_fn, wg, val_gen_kwargs, validation_generations_logger, global_step)
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Validation metrics for tau={tau_eval}: {val_metrics}")
            val_metrics["eval/temperature"] = tau_eval
            logger.log(data=val_metrics, step=global_step)

        if "swanlab" in logger.logger:
            try:
                logger.logger["swanlab"].finish()
            except Exception as e:
                pprint(f"Exception when finishing swanlab logger: {e}")


def _validate(config, tokenizer, val_dataloader, val_reward_fn, wg, val_gen_kwargs, validation_generations_logger, global_step):
    tau_eval = val_gen_kwargs.temperature
    data_source_lst = []
    reward_extra_infos_dict: dict[str, list] = defaultdict(list)

    # Lists to collect samples for the table
    sample_inputs = []
    sample_outputs = []
    sample_scores = []

    # Lists to collect lengths and sources for metric calculation
    all_response_lengths = []
    all_data_sources = []

    pending_rewards = deque()
    max_inflight = int(config.reward_model.get("async_max_inflight", 2))

    def drain_one_reward():
        future, response_lengths_cpu, data_sources = pending_rewards.popleft()
        reward_tensor, reward_result_extra_infos = ray.get(future)
        scores = reward_tensor.sum(-1).cpu().tolist()
        sample_scores.extend(scores)

        reward_extra_infos_dict["reward"].extend(scores)
        reward_extra_infos_dict["response_lengths"].extend(response_lengths_cpu.tolist())
        if reward_result_extra_infos:
            for key, lst in reward_result_extra_infos.items():
                reward_extra_infos_dict[key].extend(lst)

        data_source_lst.append(data_sources)

    for test_data in val_dataloader:

        test_batch = DataProto.from_single_dict(test_data)

        # repeat test batch
        test_batch = test_batch.repeat(repeat_times=config.data.n_samples if val_gen_kwargs.do_sample else 1, interleave=True)

        # we only do validation on rule-based rm
        if config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
            # Return a non-empty dict so callers relying on a truthy val_metrics
            # (for example, via `assert val_metrics`) do not fail, while clearly
            # indicating that validation was skipped for model-based reward models.
            return {
                "validation_skipped": True,
                "validation_reason": "model_based_reward_model",
            }

        # Store original inputs
        input_ids = test_batch.batch["input_ids"]
        # TODO: Can we keep special tokens except for padding tokens?
        input_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
        sample_inputs.extend(input_texts)

        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
        if "multi_modal_data" in test_batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("multi_modal_data")
        if "raw_prompt" in test_batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("raw_prompt")
        if "tools_kwargs" in test_batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("tools_kwargs")
        test_gen_batch = test_batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
        )

        test_gen_batch.meta_info = {
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "recompute_log_prob": False,
            "do_sample": val_gen_kwargs.do_sample,
            "temperature": val_gen_kwargs.temperature,
            "top_p": val_gen_kwargs.top_p,
            "top_k": val_gen_kwargs.top_k,
            "validate": True,
        }
        print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

        # pad to be divisible by dp_size
        test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, wg.world_size)

        test_output_gen_batch_padded = wg.generate_sequences(test_gen_batch_padded)
        # unpad
        test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
        print("validation generation end")

        # Store generated outputs
        output_ids = test_output_gen_batch.batch["responses"]
        output_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
        sample_outputs.extend(output_texts)

        test_batch = test_batch.union(test_output_gen_batch)

        # Calculate and store response lengths and their data sources
        test_batch.batch["response_mask"] = compute_response_mask(test_batch)
        response_lengths = test_batch.batch["response_mask"].sum(dim=-1)
        data_sources = test_batch.non_tensor_batch.get("data_source", ["unknown"] * len(response_lengths))
        all_response_lengths.append(response_lengths.cpu())
        all_data_sources.extend(data_sources)

        # evaluate using reward_function
        if config.reward_model.launch_reward_fn_async:
            future = compute_reward_async.remote(test_batch, config, tokenizer)
            pending_rewards.append((future, response_lengths.cpu(), data_sources))
            if len(pending_rewards) >= max_inflight:
                drain_one_reward()
        else:
            result = val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            reward_extra_infos_dict["response_lengths"].extend(response_lengths.cpu().tolist())
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)

            data_source_lst.append(data_sources)

    while pending_rewards:
        drain_one_reward()

    _maybe_log_val_generations(validation_generations_logger=validation_generations_logger, config=config, inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores, global_steps=global_step)

    # dump generations
    val_data_dir = config.trainer.get("validation_data_dir", None)
    if val_data_dir:
        _dump_generations(
            inputs=sample_inputs,
            outputs=sample_outputs,
            scores=sample_scores,
            global_steps=global_step,
            reward_extra_infos_dict=reward_extra_infos_dict,
            dump_path=val_data_dir,
        )

    for key_info, lst in reward_extra_infos_dict.items():
        assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

    data_sources = np.concatenate(data_source_lst, axis=0)

    data_src2var2metric2val = process_benchmark_metrics(data_sources, sample_inputs, reward_extra_infos_dict)
    metric_dict = {}
    for data_source, var2metric2val in data_src2var2metric2val.items():
        for var_name, metric2val in var2metric2val.items():
            for metric_name, val in metric2val.items():
                metric_dict[f"eval/{data_source}/{var_name}/{metric_name}"] = val

    # Calculate and add length-based metrics per data source
    if all_response_lengths:
        import torch

        all_response_lengths_tensor = torch.cat(all_response_lengths)
        all_data_sources_arr = np.array(all_data_sources)
        max_len = config.rollout.response_length

        for ds in np.unique(all_data_sources_arr):
            mask = all_data_sources_arr == ds
            ds_lengths = all_response_lengths_tensor[mask].float()

            if len(ds_lengths) > 0:
                metric_dict[f"eval/{ds}/response/avg_length"] = ds_lengths.mean().item()
                metric_dict[f"eval/{ds}/response/max_length"] = ds_lengths.max().item()
                cut_off_count = (ds_lengths >= max_len).sum().item()
                metric_dict[f"eval/{ds}/response/cut_off_ratio"] = cut_off_count / len(ds_lengths)

    return metric_dict


def create_rl_dataset(data_paths, data_config, tokenizer, processor):
    """Create a dataset.

    Arguments:
        data_paths: List of paths to data files.
        data_config: The data config.
        tokenizer (Tokenizer): The tokenizer.
        processor (Processor): The processor.

    Returns:
        dataset (Dataset): The dataset.
    """
    from torch.utils.data import Dataset

    from verl.utils.dataset.rl_dataset import RLHFDataset

    # Check if a custom dataset class is specified in the data configuration
    # and if the path to the custom class is provided
    if "custom_cls" in data_config and data_config.custom_cls.get("path", None) is not None:
        from verl.utils.import_utils import load_extern_type

        # Dynamically load the custom dataset class
        dataset_cls = load_extern_type(data_config.custom_cls.path, data_config.custom_cls.name)
        # Verify that the custom dataset class inherits from torch.utils.data.Dataset
        if not issubclass(dataset_cls, Dataset):
            raise TypeError(f"The custom dataset class '{data_config.custom_cls.name}' from '{data_config.custom_cls.path}' must inherit from torch.utils.data.Dataset")
    else:
        # Use the default RLHFDataset class if no custom class is specified
        dataset_cls = RLHFDataset
    print(f"Using dataset class: {dataset_cls.__name__}")

    # Instantiate the dataset using the determined dataset class
    dataset = dataset_cls(
        data_files=data_paths,
        tokenizer=tokenizer,
        processor=processor,
        config=data_config,
    )

    return dataset


def _maybe_log_val_generations(validation_generations_logger, config, inputs, outputs, scores, global_steps):
    """Log a table of validation samples to the configured logger (wandb or swanlab)"""

    generations_to_log = config.trainer.log_val_generations

    if generations_to_log == 0:
        return

    import numpy as np

    # Create tuples of (input, output, score) and sort by input text
    samples = list(zip(inputs, outputs, scores))
    samples.sort(key=lambda x: x[0])  # Sort by input text

    # Use fixed random seed for deterministic shuffling
    rng = np.random.RandomState(42)
    rng.shuffle(samples)

    # Take first N samples after shuffling
    samples = samples[:generations_to_log]

    # Log to each configured logger
    validation_generations_logger.log(config.trainer.logger, samples, global_steps)


def _dump_generations(inputs, outputs, scores, global_steps, reward_extra_infos_dict, dump_path):
    """Dump rollout/validation samples as JSONL."""
    os.makedirs(dump_path, exist_ok=True)
    filename = os.path.join(dump_path, f"{global_steps}.jsonl")

    n = len(inputs)
    base_data = {
        "input": inputs,
        "output": outputs,
        "score": scores,
        "step": [global_steps] * n,
    }

    for k, v in reward_extra_infos_dict.items():
        if len(v) == n:
            base_data[k] = v

    with open(filename, "w") as f:
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Dumped generations to {filename}")


def _load_model_path_from_hydra(run_dir: str):
    if not run_dir or not run_dir.strip():
        return None
    hydra_config_path = Path(run_dir) / ".hydra" / "config.yaml"
    if not hydra_config_path.exists():
        return None

    try:
        with hydra_config_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cfg.get("model", {}).get("path")
    except Exception:
        return None


def _load_tokenizer_with_fallback(config, run_dir: str, first_ckpt_path: str = None):
    trust_remote_code = config.data.get("trust_remote_code", False)

    candidate_paths = []
    seen_paths = set()

    def _add_candidate(path: str):
        if not path:
            return
        if path in seen_paths:
            return
        candidate_paths.append(path)
        seen_paths.add(path)

    if first_ckpt_path:
        _add_candidate(first_ckpt_path)
        parent_path = str(Path(first_ckpt_path).parent)
        if parent_path != first_ckpt_path:
            _add_candidate(parent_path)

    if run_dir:
        _add_candidate(run_dir)

    base_model_path = config.eval.get("base_model_path", None)
    if base_model_path:
        _add_candidate(base_model_path)

    if config.model.path:
        _add_candidate(config.model.path)

    hydra_model_path = _load_model_path_from_hydra(run_dir)
    if hydra_model_path:
        _add_candidate(hydra_model_path)

    attempted = []
    for raw_path in candidate_paths:
        try:
            expanded_path = str(Path(raw_path).expanduser())
            local_path = copy_to_local(expanded_path, use_shm=config.model.get("use_shm", False))
            tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
            print(f"Loaded tokenizer from: {raw_path}")
            return tokenizer
        except Exception as e:
            attempted.append(f"{raw_path}: {e}")

    raise ValueError("Unable to load tokenizer from candidates:\n" + "\n".join(attempted))


def _resolve_tau_values(eval_config) -> list[float]:
    tau_values = eval_config.get("taus")
    if tau_values:
        return [float(value) for value in tau_values]

    tau_range = eval_config.get("tau_range")
    if tau_range and tau_range.get("start") is not None:
        start = float(tau_range.start)
        end = float(tau_range.end)
        step = float(tau_range.step)
        if step <= 0:
            raise ValueError("eval.tau_range.step must be > 0")
        if end < start:
            raise ValueError("eval.tau_range.end must be >= eval.tau_range.start when step > 0")
        num_steps = int((end - start) / step) + 1
        values = [round(start + i * step, 10) for i in range(num_steps)]
        return values


    raise ValueError("Provide eval.taus or eval.tau_range.{start,end,step}.")

if __name__ == "__main__":
    main()
