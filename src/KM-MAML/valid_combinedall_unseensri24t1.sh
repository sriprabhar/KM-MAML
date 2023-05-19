MODEL='<model_name>'  
BASE_PATH='<base folder path>'

CHECKPOINT=${BASE_PATH}'/experiments/maml/'${MODEL}'/best_model.pt' #<path to where the model file is stored>
BATCH_SIZE=1
DEVICE='cuda:0'
USMASK_PATH=${BASE_PATH}'/usmasks/'

DISENTANGLE_MODEL_PATH='/<path to model file of the autoencoder trained in unsupervised model using the under-sampled images>/best_model.pt'

for DATASET_TYPE in 'sri24t1' 'sri24t2' 'sri24pd'
    do
    for MASK_TYPE in 'cartesian' 'gaussian'
        do 
	for ACC_FACTOR in '4x' '5x' '6x' '7x'  '8x' '9x'
            do 
            echo ${DATASET_TYPE}','${MASK_TYPE}','${ACC_FACTOR} 
            OUT_DIR=${BASE_PATH}'/experiments/maml/'${MODEL}'/results_unseen_zenodo/'${DATASET_TYPE}'/'${MASK_TYPE}'/acc_'${ACC_FACTOR}
            DATA_PATH=${BASE_PATH}'/datasets/'${DATASET_TYPE}'/'${MASK_TYPE}'/validation/acc_'${ACC_FACTOR}
            echo python valid.py --checkpoint ${CHECKPOINT} --out-dir ${OUT_DIR} --batch-size ${BATCH_SIZE} --device ${DEVICE} --data-path ${DATA_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} --mask_type ${MASK_TYPE} 
            python valid.py --checkpoint ${CHECKPOINT} --out-dir ${OUT_DIR} --batch-size ${BATCH_SIZE} --device ${DEVICE} --data-path ${DATA_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} --mask_type ${MASK_TYPE} --usmask_path ${USMASK_PATH} --disentangle-model-path ${DISENTANGLE_MODEL_PATH}
            done 
        done 
    done




