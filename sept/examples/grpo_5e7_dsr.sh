set -e
set -x
VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export HYDRA_FULL_ERROR=1

ROOT_DIR=$(pwd)
TRAIN_FILE=$ROOT_DIR/data/deepscaler/train.parquet

VAL_PREFIX=$ROOT_DIR/data/benchmarks
MATH500_PATH=$VAL_PREFIX/math500.parquet
AIME_PATH=$VAL_PREFIX/aime.parquet
AIME25_PATH=$VAL_PREFIX/aime25.parquet
AMC23_PATH=$VAL_PREFIX/amc23.parquet
OLYMPIAD_PATH=$VAL_PREFIX/olympiadbench.parquet
MINERVA_PATH=$VAL_PREFIX/minerva.parquet
VAL_FILE_LIST="['$MATH500_PATH', '$AMC23_PATH', '$AIME_PATH', '$AIME25_PATH', '$OLYMPIAD_PATH', '$MINERVA_PATH']"

LR=5e-7
BACKBONE="Qwen2.5-Math-7B"
BACKBONE_PATH="your_model_path"
MAX_PROMPT_LENGTH=1024
MAX_GEN_LENGTH=3072
MODEL_ID="qwen25math7b"
DATE=$(date +"%m%d_%H%M")
TASK="GRPO-noKL"
DATASET_NAME="dsr"
ROLLOUT_N=8
EXPERIMENT="PG-${DATASET_NAME}"
TAU_S=1.0
EPOCHS=3
VAL_BATCH_SIZE=256


PROJECT_NAME="formal"

mkdir -p ${ROOT_DIR}/logs
mkdir -p ${ROOT_DIR}/outputs

MODEL="${TASK}-${BACKBONE}"
EXP="${TASK}-${MODEL_ID}-${EXPERIMENT}-lr${LR}-TAUS${TAU_S}-rollout${ROLLOUT_N}-${DATE}-${EPOCHS}epoch"
OUTPUT_DIR="${ROOT_DIR}/outputs/${MODEL}/${TASK}/${EXPERIMENT}/${DATE}"


mkdir -p ${OUTPUT_DIR}

export SWANLAB_API_KEY="your_swanlab_api_key"
export SWANLAB_LOG_DIR=${ROOT_DIR}/logs/swanlab/${EXP}
export SWANLAB_MODE=cloud

mkdir -p ${SWANLAB_LOG_DIR}
LOG_FILE="${SWANLAB_LOG_DIR}/log.txt"

CUDA_VISIBLE_DEVICES=${VISIBLE_DEVICES} \
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_FILE \
    data.val_files="$VAL_FILE_LIST" \
    data.train_batch_size=128 \
    data.val_batch_size=${VAL_BATCH_SIZE} \
    data.filter_overlong_prompts=True \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_GEN_LENGTH} \
    actor_rollout_ref.model.path=${BACKBONE_PATH} \
    actor_rollout_ref.model.use_liger=False \
    actor_rollout_ref.model.use_shm=False \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.lr=${LR} \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.temperature=${TAU_S} \
    actor_rollout_ref.rollout.val_kwargs.temperature=1 \
    actor_rollout_ref.rollout.val_kwargs.n=32 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    trainer.logger=['console','swanlab'] \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXP} \
    trainer.val_before_train=True \
    trainer.default_local_dir=${OUTPUT_DIR} \
    trainer.n_gpus_per_node=8 \
    trainer.default_hdfs_dir=null \
    trainer.nnodes=1 \
    trainer.save_freq=300 \
    trainer.rollout_data_dir=${OUTPUT_DIR}/rollout_data \
    trainer.validation_data_dir=${OUTPUT_DIR}/rollout_eval_data \
    trainer.test_freq=50 \
    +trainer.log_freq=1 \
    trainer.total_epochs=${EPOCHS} | tee ${LOG_FILE}
