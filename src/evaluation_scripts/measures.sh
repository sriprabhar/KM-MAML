MODEL='<model_name>'  
BASE_PATH='<base folder path>'

CHECKPOINT=${BASE_PATH}'/'${MODEL}'/best_model.pt'
BATCH_SIZE=1
DEVICE='cuda:0'

USMASK_PATH=${BASE_PATH}'/usmasks/'

for DATASET_TYPE in 'mrbrain_t1' 'mrbrain_flair' 'ixi_pd' 'ixi_t2'
    do
    for MASK_TYPE in 'cartesian' 'gaussian'
        do 
        for ACC_FACTOR in '4x' '5x' '8x'
            do 
            echo ${DATASET_TYPE}','${MASK_TYPE}','${ACC_FACTOR} 
            DATA_PATH=${BASE_PATH}'/datasets/'${DATASET_TYPE}'/'${MASK_TYPE}'/validation/acc_'${ACC_FACTOR}
            REPORT_PATH=${BASE_PATH}'/experiments/'${MODEL}'/reports_for_seen/'
            echo python valid_measures_csv.py --checkpoint ${CHECKPOINT} --batch-size ${BATCH_SIZE} --device ${DEVICE} --data-path ${DATA_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} --usmask_path ${USMASK_PATH} --report-path ${REPORT_PATH}
            python valid_measures_csv.py --checkpoint ${CHECKPOINT} --batch-size ${BATCH_SIZE} --device ${DEVICE} --data-path ${DATA_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} --mask_type ${MASK_TYPE} --usmask_path ${USMASK_PATH} --report-path ${REPORT_PATH}
            done 
        done 
    done



