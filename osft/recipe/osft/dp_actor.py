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

import logging
import os
import torch

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, kl_penalty
from verl.utils.debug import GPUMemoryLogger
from verl.utils.device import get_device_id, is_cuda_available, is_npu_available
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import rearrange_micro_batches
from verl.workers.actor import DataParallelPPOActor

if is_cuda_available:
    pass
elif is_npu_available:
    pass


__all__ = ["DataParallelPPOActor"]


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class OSFTDataParallelPPOActor(DataParallelPPOActor):
    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        multi_turn = data.meta_info.get("multi_turn", False)

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        if multi_turn:
            select_keys.append("loss_mask")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        # usually it's 1
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                # split batch into micro_batches
                mini_batch = data
                if has_multi_modal_inputs:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    # split batch into micro_batches
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for data in micro_batches:
                    # Support all hardwares
                    if isinstance(data, DataProto):
                        data = {**data.batch.to(get_device_id()), **data.non_tensor_batch}
                    else:
                        data = data.to(get_device_id())  # actor device is cpu when using offload
                    responses = data["responses"]
                    response_length = responses.size(1)
                    attention_mask = data["attention_mask"]
                    if multi_turn:
                        response_mask = data["loss_mask"][:, -response_length:]
                    else:
                        response_mask = attention_mask[:, -response_length:]

                    loss_agg_mode = self.config.loss_agg_mode
                    policy_loss_type = str(self.config.get("policy_loss_type", "cross_entropy")).lower()
                    entropy_loss_coeff = float(self.config.get("entropy_loss_coeff", 1.0))

                    if policy_loss_type not in {"cross_entropy", "entropy_minimization"}:
                        raise ValueError(
                            f"Unsupported policy_loss_type: {policy_loss_type}. "
                            "Expected one of: cross_entropy, entropy_minimization."
                        )

                    calculate_entropy = policy_loss_type == "entropy_minimization"

                    # we should use log_prob to calculate the cross entropy loss
                    entropy, log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature, calculate_entropy=calculate_entropy)

                    # Cross-Entropy Loss is equivalent to Negative Log-Likelihood.
                    cross_entropy_loss = agg_loss(loss_mat=-log_prob, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                    policy_loss = cross_entropy_loss
                    entropy_loss = None
                    if calculate_entropy:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        policy_loss = entropy_loss * entropy_loss_coeff

                    with torch.no_grad():
                        masked_log_prob = log_prob * response_mask
                        num_tokens = response_mask.sum()
                        if num_tokens > 0:
                            mean_log_prob = masked_log_prob.sum() / num_tokens
                            perplexity = torch.exp(-mean_log_prob)
                        else:
                            # Avoid division by zero if there are no valid tokens.
                            perplexity = torch.tensor(0.0, device=log_prob.device)

                    if self.config.use_kl_loss:
                        ref_log_prob = data["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type)
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] = kl_loss.detach().item()
                        metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * (len(data) / self.config.ppo_mini_batch_size)
                    else:
                        loss = policy_loss / self.gradient_accumulation
                    loss.backward()

                    data = {
                        "actor/pg_loss": policy_loss.detach().item(), # here pg is just for alignment with grpo's logging
                        "actor/cross_entropy_loss": cross_entropy_loss.detach().item(),
                        "actor/perplexity": perplexity.detach().item(),
                    }
                    if entropy_loss is not None:
                        data["actor/entropy_loss"] = entropy_loss.detach().item()
                    append_to_dict(metrics, data)

                grad_norm = self._optimizer_step()
                data = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, data)
        self.actor_optimizer.zero_grad()
        return metrics
