MODEL='<model_name>'  
BASE_PATH='<base folder path>'

DATA_DIR='<path to data folder>'

RESULTS_TYPE='unadapted'
EXPERIMENT_PATH=${BASE_PATH}'/experiments/'

for DATASET_TYPE in 'sri24t1' 'sri24t2' 'sri24pd'
    do
    for MASK_TYPE in 'cartesian' 'gaussian'
        do 
        for ACC_FACTOR in '6x' '7x' '9x' '4x' '5x' '8x'
            do 
            echo ${DATASET_TYPE}','${MASK_TYPE}','${ACC_FACTOR} 
            DATA_PATH=${BASE_PATH}'/datasets/'${DATASET_TYPE}'/'${MASK_TYPE}'/validation/acc_'${ACC_FACTOR}

	    PREDICTIONS_PATH=${EXPERIMENT_PATH}'/'${MODEL}'/results_for_unseen/'${RESULTS_TYPE}'/'${DATASET_TYPE}'/'${MASK_TYPE}'/acc_'${ACC_FACTOR}
            REPORT_PATH=${EXPERIMENT_PATH}'/'${MODEL}'/reports_for_unseen/'
            echo python valid_measures_csv.py --data-path ${DATA_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} --mask_type ${MASK_TYPE} --usmask_path ${USMASK_PATH} --report-path ${REPORT_PATH}
            python valid_measures_csv.py --target-path ${DATA_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} --mask_type ${MASK_TYPE} --report-path ${REPORT_PATH} --predictions-path ${PREDICTIONS_PATH}
            done 
        done 
    done




