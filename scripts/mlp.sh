###
 # @Author: Jiaxin Zheng
 # @Date: 2024-03-09 12:32:00
 # @LastEditors: Jiaxin Zheng
 # @LastEditTime: 2024-08-16 11:52:54
 # @Description: 1 task 0.5h 180tasks 90h  25h
### 
export CUDA_VISIBLE_DEVICES=0
export LOG_DIR='logs'

LR=1e-4
BATCH_SIZE=128
MAX_EPOCHS=50
DROPOUT_RATE=0.0
WARMUP=0.02

FEATURE_DIR='data/model_feature'
DATA_DIR='data/labels'
SPLIT_FILE='data/data.csv'

FEATURE_FILE_LIST=(
    "mols_unimol.pkl"
)

TASK_NAME_LIST=(
    "docking_score"
)

TARGET_LIST=(
    "ADRB2"
)

for FEATURE_FILE_IDX in "${!FEATURE_FILE_LIST[@]}"; do
    FILE=${FEATURE_FILE_LIST[${FEATURE_FILE_IDX}]}    
    FEATURE_NAME="${FILE%.pkl}"
    EMBED_PATH=${FEATURE_DIR}//${FILE}

    for TARGET in "${TARGET_LIST[@]}"; do
        
        # for TASK_NAME in "${TASK_NAME_LIST[@]}"; do
        for TASK_IDX in "${!TASK_NAME_LIST[@]}"; do
            TASK_NAME=${TASK_NAME_LIST[${TASK_IDX}]} 

            # TASK_IDX=${TASK_DICT[$TASK_NAME]}

            python src/mlp_main.py \
                experiment=moleculecla_mlp \
                tags=\["${FEATURE_NAME}","${TASK_NAME}"\] \
                task_name=${FEATURE_NAME}_${TARGET}_${TASK_NAME} \
                data.task_idx=${TASK_IDX} \
                data.train.dataset.target=${TARGET} \
                data.val.dataset.target=${TARGET} \
                data.test.dataset.target=${TARGET} \
                data.train.dataset.embed_path=${EMBED_PATH} \
                data.val.dataset.embed_path=${EMBED_PATH} \
                data.test.dataset.embed_path=${EMBED_PATH} \
                model.optimizer.lr=${LR} \
                data.batch_size=${BATCH_SIZE} \
                trainer.max_epochs=${MAX_EPOCHS} \
                model.model.dropout_rate=${DROPOUT_RATE} \
                model.lr_scheduler.warmup=${WARMUP}
        done
    done
done