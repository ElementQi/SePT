set -e
set -x

VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
nproc_per_node=8
export HYDRA_FULL_ERROR=1

ROOT_DIR=$(pwd)
RUN_TAU="tau0s6"

TRAIN_FILE=$ROOT_DIR/outputs/generated_solutions/$RUN_TAU/train.parquet
VAL_FILE=$ROOT_DIR/outputs/generated_solutions/$RUN_TAU/test.parquet

MODEL="qwen25math7b"
TASK="offline-sept"
MODEL_PATH="your_model_path"

LR=1e-5
EPOCHS=3

RUN_NAME="dsr_${RUN_TAU}_lr${LR}_epoch${EPOCHS}"
PROJECT_NAME="offline-sept"
DATE=$(date +"%m%d_%H%M")

OUTPUT_DIR="${ROOT_DIR}/outputs/${MODEL}/${TASK}/lr${LR}-${EPOCHS}epochs-${RUN_TAU}/${DATE}"
EXP="${MODEL}/${TASK}/lr${LR}-${EPOCHS}epochs-${RUN_TAU}/${DATE}"

mkdir -p ${OUTPUT_DIR}

export SWANLAB_API_KEY="your_swanlab_api_key"
export SWANLAB_LOG_DIR=${ROOT_DIR}/logs/swanlab/${EXP}
export SWANLAB_MODE=cloud

mkdir -p ${SWANLAB_LOG_DIR}
LOG_FILE="${SWANLAB_LOG_DIR}/log.txt"

CUDA_VISIBLE_DEVICES=${VISIBLE_DEVICES} \
torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$TRAIN_FILE \
    data.val_files=$VAL_FILE \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    data.prompt_dict_keys=['question'] \
    data.response_dict_keys=['answer'] \
    data.micro_batch_size_per_gpu=4 \
    data.max_length=4096 \
    data.train_batch_size=128 \
    model.partial_pretrain=$MODEL_PATH \
    model.use_shm=False \
    model.fsdp_config.model_dtype=bf16 \
    optim.lr=$LR \
    optim.warmup_steps_ratio=0.01 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=5 \
    trainer.default_local_dir=$OUTPUT_DIR \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$RUN_NAME \
    trainer.total_epochs=$EPOCHS \
    +trainer.log_freq=1 \
    trainer.logger=['console','swanlab'] \
    trainer.default_hdfs_dir=null \
    "$@" 2>&1 | tee ${LOG_FILE}
