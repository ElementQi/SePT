#!/bin/bash
# Generate solutions from the DeepScaler training set at temperatures 0.6
# using Qwen/Qwen2.5-Math-7B as the base model.
#
# One output parquet per temperature is written to OUTPUT_DIR:
#   solutions_tau0s6.parquet   (temperature 0.6)
#
# Each file contains the original dataset columns plus a `solution` column.

set -e
set -x

VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export HYDRA_FULL_ERROR=1

ROOT_DIR=$(pwd)
TRAIN_FILE=$ROOT_DIR/data/deepscaler/train.parquet

MODEL_PATH="your_model_path"
OUTPUT_DIR="${ROOT_DIR}/outputs/generated_solutions"

MAX_PROMPT_LENGTH=1024
MAX_GEN_LENGTH=3072

mkdir -p "${OUTPUT_DIR}"

CUDA_VISIBLE_DEVICES=${VISIBLE_DEVICES} \
python3 -m recipe.sept.generate_solutions \
    data.path=${TRAIN_FILE} \
    data.output_dir=${OUTPUT_DIR} \
    data.filter_overlong_prompts=True \
    data.batch_size=256 \
    model.path=${MODEL_PATH} \
    rollout.prompt_length=${MAX_PROMPT_LENGTH} \
    rollout.response_length=${MAX_GEN_LENGTH} \
    rollout.tensor_model_parallel_size=1 \
    rollout.gpu_memory_utilization=0.9 \
    rollout.enforce_eager=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    "eval.taus=[0.6]"
