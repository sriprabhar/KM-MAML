MODEL='<model_name>'  
BASE_PATH='<base folder path>'

echo ${MODEL}

EXPERIMENT_PATH=${BASE_PATH}'/experiments/maml'

BASE_REPORT_PATH=${EXPERIMENT_PATH}'/'${MODEL}

EVALUATE_TASK_STRINGS='mrbrain_t1_few_gaussian_4x','mrbrain_t1_few_gaussian_5x','mrbrain_t1_few_gaussian_8x','mrbrain_t1_few_cartesian_4x','mrbrain_t1_few_cartesian_5x','mrbrain_t1_few_cartesian_8x','mrbrain_flair_few_gaussian_4x','mrbrain_flair_few_gaussian_5x','mrbrain_flair_few_gaussian_8x','mrbrain_flair_few_cartesian_4x','mrbrain_flair_few_cartesian_5x','mrbrain_flair_few_cartesian_8x'



for DATASET_TYPE in 'mrbrain_t1_few' 'mrbrain_flair_few'
    do
    for MASK_TYPE in 'cartesian' 'gaussian'
        do 
        for ACC_FACTOR in '4x' '5x' '8x'
            do 
            echo ${DATASET_TYPE}','${MASK_TYPE}','${ACC_FACTOR} 
            REPORT_PATH=${BASE_REPORT_PATH}'/report_'${DATASET_TYPE}'_'${MASK_TYPE}'_'${ACC_FACTOR}'.txt'
            echo ${REPORT_PATH}
            cat ${REPORT_PATH}
            echo "\n"
            done 
        done 
    done 

