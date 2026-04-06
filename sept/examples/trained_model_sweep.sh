set -x

export HYDRA_FULL_ERROR=1

MODEL_PATH="your_base_model_path"
CKPT_PATH="your_ckpt_path"
BACKBONE="your_ckpt_name"

VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

ROOT_DIR=$(pwd)
VAL_PREFIX=$ROOT_DIR/data/benchmarks
MATH500_PATH=$VAL_PREFIX/math500.parquet
AIME_PATH=$VAL_PREFIX/aime.parquet
AIME25_PATH=$VAL_PREFIX/aime25.parquet
AMC23_PATH=$VAL_PREFIX/amc23.parquet
OLYMPIAD_PATH=$VAL_PREFIX/olympiadbench.parquet
MINERVA_PATH=$VAL_PREFIX/minerva.parquet
VAL_FILE_LIST="['$MATH500_PATH', '$AMC23_PATH', '$AIME_PATH', '$AIME25_PATH', '$OLYMPIAD_PATH', '$MINERVA_PATH']"

MAX_PROMPT_LENGTH=1024
MAX_GEN_LENGTH=3072

N=32
TAU_START=0
TAU_END=1.6
TAU_STEP=0.1
BATCH_SIZE=256

TASK="ckpt-tau-sweep"
PROJECT_NAME="ckpt-tau-sweep"
DATE=$(date +"%m%d_%H%M")
MODEL="${BACKBONE}"
OUTPUT_DIR="${ROOT_DIR}/outputs/ckpt_tau_sweep/${MODEL}/${DATE}"

mkdir -p ${OUTPUT_DIR}
mkdir -p ${OUTPUT_DIR}/logs

EXP="${TASK}-${MODEL}-${DATE}-n${N}-tau${TAU_START}-${TAU_END}-step${TAU_STEP}"
LOG_FILE="${OUTPUT_DIR}/logs/${EXP}.log"
export SWANLAB_API_KEY="your_swanlab_api_key"
export SWANLAB_LOG_DIR=${ROOT_DIR}/logs/swanlab/${EXP}
export SWANLAB_MODE=cloud
mkdir -p ${SWANLAB_LOG_DIR}

CUDA_VISIBLE_DEVICES=${VISIBLE_DEVICES} \
python3 -m recipe.sept.tau_sweep \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    trainer.validation_data_dir=${OUTPUT_DIR} \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXP} \
    eval.taus=null \
    eval.tau_range.start=${TAU_START} \
    eval.tau_range.end=${TAU_END} \
    eval.tau_range.step=${TAU_STEP} \
    data.val_files="$VAL_FILE_LIST" \
    data.prompt_key=prompt \
    data.n_samples=${N} \
    data.batch_size=${BATCH_SIZE} \
    model.path=${MODEL_PATH} \
    +eval.ckpt_path=${CKPT_PATH} \
    model.use_shm=False \
    +model.trust_remote_code=False \
    reward_model.launch_reward_fn_async=True \
    rollout.temperature=1 \
    rollout.top_k=-1 \
    rollout.top_p=1 \
    rollout.prompt_length=${MAX_PROMPT_LENGTH} \
    rollout.response_length=${MAX_GEN_LENGTH} \
    rollout.tensor_model_parallel_size=1 \
    rollout.gpu_memory_utilization=0.9
