MODEL='<model name>'
BASE_PATH='<base folder name>'
MASK_TYPE='cartesian','gaussian'
DATASET_TYPE='mrbrain_t1','kneeMRI320x320'
ACC_FACTORS='4x','5x','8x'
BATCH_SIZE=4
NUM_EPOCHS=150
DEVICE='cuda:0'
EXP_DIR=${BASE_PATH}'/experiments/'${MODEL}
TRAIN_PATH=${BASE_PATH}'/datasets/' # train data path h5 files
VALIDATION_PATH=${BASE_PATH}'/datasets/' # validation data path h5 files
USMASK_PATH=${BASE_PATH}'/usmasks/'
echo python train.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --dataset_type ${DATASET_TYPE} --usmask_path ${USMASK_PATH} --acceleration_factor ${ACC_FACTORS} --mask_type ${MASK_TYPE}
python train.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --dataset_type ${DATASET_TYPE} --usmask_path ${USMASK_PATH} --acceleration_factor ${ACC_FACTORS} --mask_type ${MASK_TYPE}
