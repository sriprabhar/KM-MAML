MODEL='<model_name>'  
BASE_PATH='<base folder path>'



for DATASET_TYPE in 'sri24t1' 'sri24t2' 'sri24pd'
    do
    for MASK_TYPE in 'cartesian' 'gaussian'
        do 
        for ACC_FACTOR in '4x' '5x' '6x' '7x'  '8x' '9x'
            do 
            echo ${DATASET_TYPE}','${MASK_TYPE}','${ACC_FACTOR} 
            TARGET_PATH=${BASE_PATH}'/datasets/'${DATASET_TYPE}'/'${MASK_TYPE}'/validation/acc_'${ACC_FACTOR}
            PREDICTIONS_PATH=${BASE_PATH}'/experiments/sgd/combined_dataset_mask_acc_factor_for_decouplelearning/'${MODEL}'/results_unseen_zenodo/'${DATASET_TYPE}'/'${MASK_TYPE}'/acc_'${ACC_FACTOR}'/'
            REPORT_PATH=${BASE_PATH}'/experiments/sgd/combined_dataset_mask_acc_factor_for_decouplelearning/'${MODEL}'/'
            python evaluate_slicewise.py --target-path ${TARGET_PATH} --predictions-path ${PREDICTIONS_PATH} --report-path ${REPORT_PATH} --acc-factor ${ACC_FACTOR} --mask-type ${MASK_TYPE} --dataset-type ${DATASET_TYPE}
            done 
        done 
    done
