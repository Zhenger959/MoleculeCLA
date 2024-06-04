###
 # @Author: Jiaxin Zheng
 # @Date: 2024-03-09 12:32:00
 # @LastEditors: Jiaxin Zheng
 # @LastEditTime: 2024-06-04 10:22:48
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
    "unimol_plus_pcq_large_dockingdata.pkl"
    "Frad_feats.pkl"
    "3d_denoising_feats.pkl"
    "Slide_feats.pkl"
)

TASK_NAME_LIST=(
    "docking_score"
    "glide_lipo"
    "glide_hbond"
    "glide_evdw"
    "glide_ecoul"
    "glide_erotb"
    "glide_esite"
    "glide_emodel"
    "glide_einternal"
)
# ! same order ['docking_score', 'glide_lipo', 'glide_hbond', 'glide_evdw', 'glide_ecoul', 'glide_erotb', 'glide_esite', 'glide_emodel', 'glide_einternal']
TARGET_LIST=(
    "ADRB2"
    "ABL1"
    "CYT2C9"
    "PPARG"
    "GluA2"
    "3CL"
    "HIVINT"
    "HDAC2"
    "KRAS"
    "PDE5"
)

for FEATURE_FILE_IDX in "${!FEATURE_FILE_LIST[@]}"; do
    FILE=${FEATURE_FILE_LIST[${FEATURE_FILE_IDX}]}    
    FEATURE_NAME="${FILE%.pkl}"
    EMBED_PATH=${FEATURE_DIR}//${FILE}

    for TARGET in "${TARGET_LIST[@]}"; do
        
        for TASK_NAME in "${TASK_NAME_LIST[@]}"; do

            python src/mlp_main.py \
                experiment=moleculecla_mlp \
                project=moleculecla \
                group=${FEATURE_NAME} \
                tags=\["${FEATURE_NAME}","${TASK_NAME}"\] \
                task_name=${FEATURE_NAME}_${TARGET}_${TASK_NAME} \
                data.task_idx=${TASK_IDX} \
                data.train.dataset.target=${TARGET} \
                data.val.dataset.target=${TARGET} \
                data.test.dataset.target=${TARGET} \
                data.train.dataset.embed_path=${EMBED_PATH} \
                data.val.dataset.embed_path=${EMBED_PATH} \
                data.test.dataset.embed_path=${EMBED_PATH} \
                data.datadir=${DATA_DIR} \
                data.train.dataset.datadir=${DATA_DIR} \
                data.val.dataset.datadir=${DATA_DIR} \
                data.test.dataset.datadir=${DATA_DIR} \
                data.train.dataset.split_file=${SPLIT_FILE} \
                data.val.dataset.split_file=${SPLIT_FILE} \
                data.test.dataset.split_file=${SPLIT_FILE} \
                model.optimizer.lr=${LR} \
                data.batch_size=${BATCH_SIZE} \
                trainer.max_epochs=${MAX_EPOCHS} \
                model.model.dropout_rate=${DROPOUT_RATE} \
                model.lr_scheduler.warmup=${WARMUP}
        done
    done
done