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
"""Evaluate FSDP checkpoints at 50-step intervals using rollout validation."""

import json
import os
import re
from pathlib import Path
from pprint import pprint

import hydra
import ray
from omegaconf import OmegaConf
from torchdata.stateful_dataloader import StatefulDataLoader

from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.trainer.ppo.reward import load_reward_manager
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.fsdp_workers import ActorRolloutRefWorker

from .tau_sweep import _load_model_path_from_hydra, _load_tokenizer_with_fallback, _validate, create_rl_dataset

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"


@hydra.main(config_path="config", config_name="osft_generation", version_base=None)
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

        from verl.utils.tracking import Tracking

        run_dir = _resolve_run_dir(config)
        ckpts = _discover_checkpoints(run_dir)
        if not ckpts:
            raise ValueError(f"No eligible checkpoints (step % 50 == 0) found under run_dir={run_dir}")

        # Get first checkpoint to use for loading weights
        first_ckpt_step, first_ckpt_path = ckpts[0]
        
        # Load tokenizer with fallback options including checkpoint directories
        tokenizer = _load_tokenizer_with_fallback(config, run_dir, first_ckpt_path)

        # Ensure config.model.path points to a valid HuggingFace model directory for initialization
        # (not the FSDP checkpoint directory which contains only sharded .pt files)
        model_path_valid = False
        if config.model.path:
            try:
                model_path_valid = Path(config.model.path).expanduser().exists()
            except Exception as e:
                # Invalid path format or inaccessible path
                pprint(f"Warning: config.model.path is invalid or inaccessible: {e}")
        
        if not model_path_valid:
            # Try to get the original model path from hydra config
            hydra_model_path = _load_model_path_from_hydra(run_dir)
            if hydra_model_path:
                config.model.path = hydra_model_path
            elif config.eval.get("base_model_path"):
                config.model.path = config.eval.base_model_path
            else:
                path_display = config.model.path if config.model.path else "not set"
                raise ValueError(
                    f"config.model.path ({path_display}) is not set to a valid HuggingFace model directory. "
                    f"Please set model.path to the base model directory or use +eval.base_model_path=<path> "
                    f"to specify the original model used for training."
                )
        
        ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(ActorRolloutRefWorker), config=config, role="rollout")
        resource_pool = RayResourcePool(process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes, use_gpu=True, max_colocate_count=1)
        wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init, device_name=config.trainer.device)
        
        # Initialize model structure with base model
        wg.init_model()
        
        # Load the first checkpoint weights
        wg.load_checkpoint(local_path=first_ckpt_path, hdfs_path=None, del_local_after_load=False)

        val_dataset = create_rl_dataset(config.data.get("val_files", config.data.path), config.data, tokenizer, None)
        val_bs_cfg = config.data.get("batch_size", None)
        val_batch_size = len(val_dataset) if val_bs_cfg is None or int(val_bs_cfg) <= 0 else min(int(val_bs_cfg), len(val_dataset))
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

        validation_generations_logger = ValidationGenerationsLogger()
        logger = Tracking(
            project_name=config.trainer.project_name,
            experiment_name=config.trainer.experiment_name,
            default_backend=config.trainer.logger,
            config=OmegaConf.to_container(config, resolve=True),
        )

        config.data.n_samples = config.eval.get("n_samples", 32)

        # Allow generation parameters to be overridden via eval config while keeping
        # current behavior as the default.
        eval_gen_cfg = config.eval.get("generation", {})
        temperature = eval_gen_cfg.get("temperature", 1.0)
        top_p = eval_gen_cfg.get("top_p", config.rollout.top_p)
        top_k = eval_gen_cfg.get("top_k", config.rollout.top_k)
        do_sample = eval_gen_cfg.get("do_sample", True)

        val_gen_kwargs = OmegaConf.create(
            {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "do_sample": do_sample,
            }
        )

        output_jsonl = Path(run_dir) / "ckpt_eval_results.jsonl"
        summary_rows = []

        with output_jsonl.open("w", encoding="utf-8") as fout:
            for ckpt_step, ckpt_path in ckpts:
                record = {"step": ckpt_step, "ckpt_path": ckpt_path}
                try:
                    # Load checkpoint using load_checkpoint instead of reinitializing
                    wg.load_checkpoint(local_path=ckpt_path, hdfs_path=None, del_local_after_load=False)
                    metric_dict = _validate(config, tokenizer, val_dataloader, val_reward_fn, wg, val_gen_kwargs, validation_generations_logger, ckpt_step)
                    record["metrics"] = metric_dict
                    logger.log(data=metric_dict, step=ckpt_step)
                    summary_rows.append((ckpt_step, "ok", metric_dict.get("eval/all/score/mean", "-")))
                    pprint(f"Validation metrics for ckpt step={ckpt_step}: {metric_dict}")
                except Exception as e:
                    record["error"] = str(e)
                    summary_rows.append((ckpt_step, "failed", str(e)))
                    pprint(f"Failed checkpoint step={ckpt_step}, path={ckpt_path}, err={e}")
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()

        print(f"Saved checkpoint evaluation results to {output_jsonl}")
        _print_summary(summary_rows)

        if "swanlab" in logger.logger:
            try:
                logger.logger["swanlab"].finish()
            except Exception as e:
                pprint(f"Exception when finishing swanlab logger: {e}")


def _resolve_run_dir(config) -> str:
    run_dir = config.eval.get("run_dir", None)
    if run_dir:
        return str(Path(run_dir).expanduser())
    return str(Path(config.model.path).expanduser())


def _discover_checkpoints(run_dir: str) -> list[tuple[int, str]]:
    run_path = Path(run_dir).expanduser()
    if not run_path.is_dir():
        raise ValueError(f"run_dir does not exist or is not a directory: {run_path}")

    ckpts: list[tuple[int, str]] = []
    for child in run_path.iterdir():
        if not child.is_dir():
            continue
        step = _parse_step_from_name(child.name)
        if step is None or step % 50 != 0:
            continue

        # HuggingFace safetensors checkpoints (offline SFT) are stored directly
        # under the step directory without an 'actor' sub-directory.
        if _is_hf_safetensors_checkpoint(child):
            ckpts.append((step, str(child)))
            continue

        # For FSDP checkpoints, look for actor subdirectory
        actor_dir = child / "actor"
        if actor_dir.is_dir():
            # Validate that this looks like an FSDP checkpoint
            if _validate_fsdp_checkpoint(actor_dir):
                ckpts.append((step, str(actor_dir)))
        elif _looks_like_complete_ckpt(child):
            # Fallback for checkpoints that don't have actor subdirectory
            ckpts.append((step, str(child)))

    ckpts.sort(key=lambda x: x[0])
    return ckpts


def _parse_step_from_name(name: str):
    patterns = [
        r"global_step[-_]?([0-9]+)",
        r"checkpoint[-_]?([0-9]+)",
        r"step[-_]?([0-9]+)",
        r"(?:^|[-_])([0-9]+)$",
    ]
    for pattern in patterns:
        m = re.search(pattern, name)
        if m:
            return int(m.group(1))
    return None


def _validate_fsdp_checkpoint(ckpt_dir: Path) -> bool:
    """
    Validate that a directory contains FSDP checkpoint shard files.
    
    FSDP checkpoints should have files named like:
    - model_world_size_<N>_rank_<R>.pt (model shards)
    - config.json (model config)
    """
    if not ckpt_dir.is_dir():
        return False
    
    # Check for at least one model shard file
    has_model_shard = any(
        f.name.startswith("model_world_size_") and f.name.endswith(".pt")
        for f in ckpt_dir.iterdir()
        if f.is_file()
    )
    
    # Check for config.json which should be present for HF-style checkpoints
    has_config = (ckpt_dir / "config.json").exists()
    
    return has_model_shard and has_config


def _is_hf_safetensors_checkpoint(ckpt_dir: Path) -> bool:
    """Return True if *ckpt_dir* is a HuggingFace safetensors checkpoint.

    Recognised by the presence of ``model.safetensors`` (single-shard) or
    ``model.safetensors.index.json`` (multi-shard) in the directory.
    """
    return (ckpt_dir / "model.safetensors.index.json").exists() or (ckpt_dir / "model.safetensors").exists()


def _looks_like_complete_ckpt(ckpt_dir: Path) -> bool:
    # Heuristic: directory has any non-hidden file or nested file.
    # Broken/incomplete directories are skipped and exceptions are still handled per-ckpt.
    for item in ckpt_dir.iterdir():
        if item.name.startswith("."):
            continue
        if item.is_file():
            return True
        if item.is_dir() and any(not p.name.startswith(".") for p in item.iterdir()):
            return True
    return False


def _print_summary(rows):
    print("\nCheckpoint sweep summary (sorted by step):")
    print(f"{'step':>8}  {'status':>8}  detail")
    for step, status, detail in sorted(rows, key=lambda x: x[0]):
        print(f"{step:>8}  {status:>8}  {detail}")


if __name__ == "__main__":
    main()
