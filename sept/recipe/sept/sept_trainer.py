# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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

import uuid
import math
from pprint import pprint
from typing import Optional, Type

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm

from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.ppo.metric_utils import (
    compute_throughout_metrics,
    compute_timing_metrics,
)
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager, Role, compute_response_mask
from verl.utils.debug.performance import _timer
from verl.utils.metric import (
    reduce_metrics,
)
from verl.utils.tracking import ValidationGenerationsLogger

WorkerType = Type[Worker]


class RaySePTTrainer(RayPPOTrainer):
    """
    RaySePTTrainer is a trainer for Self-Evolving Post-Training (SePT) using Ray.
    """

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name="cuda",
    ):
        """Initialize distributed PPO trainer with Ray backend."""

        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn
        self.use_critic = False

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.ref_in_actor = config.actor_rollout_ref.model.get("lora_rank", 0) > 0

        self.use_rm = False
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name
        self.validation_generations_logger = ValidationGenerationsLogger()

        self._validate_config()
        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def fit(self):
        """
        The training loop of SePT.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the SePT dataflow.
        """

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # pop those keys for generation
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
                if "multi_modal_data" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )

                # anneal tau_s if needed
                current_tau_s = self.config.actor_rollout_ref.rollout.temperature
                anneal_config = self.config.actor_rollout_ref.rollout.get("tau_s_annealing", {})

                if anneal_config.get("enable", False):
                    start_tau = float(anneal_config.get("start_tau", 0.3))
                    end_tau = float(anneal_config.get("end_tau", 0.6))
                    anneal_epochs = int(anneal_config.get("epochs", 2))
                    anneal_mode = str(anneal_config.get("mode", "linear")).lower()

                    # Optional: allow overriding the number of steps directly
                    total_anneal_steps = anneal_config.get("steps")
                    if total_anneal_steps is None:
                        steps_per_epoch = max(1, self.total_training_steps // max(1, self.config.trainer.total_epochs))
                        total_anneal_steps = max(1, anneal_epochs * steps_per_epoch)

                    current_step = max(0, self.global_steps - 1)  # zero-based
                    if current_step < total_anneal_steps:
                        progress = current_step / total_anneal_steps  # in [0, 1)
                        if anneal_mode == "linear":
                            # Linear: start -> end at a constant rate
                            current_tau_s = start_tau + (end_tau - start_tau) * progress
                        else:
                            # Cosine: smooth start -> end
                            cosine_value = 0.5 * (1 + math.cos(math.pi * progress))
                            current_tau_s = end_tau - (end_tau - start_tau) * cosine_value
                    else:
                        current_tau_s = end_tau

                    # Update the temperature for the rollout
                    gen_batch.meta_info["temperature"] = current_tau_s
                    gen_batch.meta_info["taus_anneal"] = True
                    metrics["train/current_tau_s"] = float(current_tau_s)
                    metrics["train/taus_anneal_progress"] = min(1.0, current_step / total_anneal_steps)

                is_last_step = self.global_steps >= self.total_training_steps

                with _timer("step", timing_raw):
                    # generate a batch
                    with _timer("gen", timing_raw):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        else:
                            self.async_rollout_manager.wake_up()
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                            self.async_rollout_manager.sleep()
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)

                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    batch.batch["response_mask"] = compute_response_mask(batch)

                    # compute scores using function-based reward 'model'
                    if self.config.trainer.enable_train_reward and self.reward_fn is not None:
                        reward_tensor = self.reward_fn(batch)
                        batch.batch["token_level_scores"] = reward_tensor
                        # we do not have adv, therefore we use scores as rewards
                        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # Use a single tau_t source for both ref-logprob and actor loss.
                    # This keeps KL temperature-consistent when train temperature is enabled,
                    # while preserving vanilla behavior (tau_t = 1.0) otherwise.
                    train_temperature = current_tau_s if self.config.trainer.enable_train_temperature else 1.0
                    batch.meta_info["temperature"] = train_temperature

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer("ref", timing_raw):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob_tau_t(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob_tau_t(batch)
                            batch = batch.union(ref_log_prob)

                    # this is used to indicate the temperature for loss calculation.
                    batch.meta_info["temperature"] = train_temperature

                    # update actor (core part)
                    with _timer("update_actor", timing_raw):
                        batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                        actor_output = self.actor_rollout_wg.update_actor(batch)
                    actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                    metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with _timer("dump_rollout_generations", timing_raw):
                            # print(batch.batch.keys())
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            # scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()

                            # --- Scores and Rewards (from reward_fn) ---
                            if "token_level_scores" in batch.batch and batch.batch["token_level_scores"] is not None:
                                sequence_score = batch.batch["token_level_scores"].sum(-1)
                                scores = sequence_score.cpu().tolist()
                                metrics.update(
                                    {
                                        "reward/score/mean": torch.mean(sequence_score).item(),
                                        "reward/score/max": torch.max(sequence_score).item(),
                                        "reward/score/min": torch.min(sequence_score).item(),
                                    }
                                )
                            else:
                                print("DEBUG dump_rollout_generations: 'token_level_scores' not found.")
                                scores = [0 for _ in range(len(inputs))]  # placeholder, since we don't have scores in SePT

                            response_lengths = batch.batch["response_mask"].sum(dim=-1).cpu().tolist()

                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                reward_extra_infos_dict={"response_lengths": response_lengths},
                                dump_path=rollout_data_dir,
                            )

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics

                # no reward_fn, so no reward metrics from compute_data_metrics
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1
                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

    def _validate_base(self):
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None:
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)

        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            import json
            json.dump(val_metrics, open(f"{val_data_dir}/the_sub_metric.json", "w"), indent=4)

        if "swanlab" in logger.logger:
            logger.logger["swanlab"].finish()
        return val_metrics
