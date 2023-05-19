MODEL='<model_name>'  
BASE_PATH='<base folder path>'

echo ${MODEL}


EXPERIMENT_PATH=${BASE_PATH}'/experiments/maml/'

BASE_REPORT_PATH=${EXPERIMENT_PATH}'/'${MODEL}'/results_unseen_adapt_30'

#EVALUATE_TASK_STRINGS='mrbrain_t1_few_gaussian_4x','mrbrain_t1_few_gaussian_5x','mrbrain_t1_few_gaussian_8x','mrbrain_t1_few_cartesian_4x','mrbrain_t1_few_cartesian_5x','mrbrain_t1_few_cartesian_8x','mrbrain_flair_few_gaussian_4x','mrbrain_flair_few_gaussian_5x','mrbrain_flair_few_gaussian_8x','mrbrain_flair_few_cartesian_4x','mrbrain_flair_few_cartesian_5x','mrbrain_flair_few_cartesian_8x'


EVALUATE_TASK_STRINGS='sri24_pd_few_cartesian_4x','sri24_pd_few_cartesian_5x','sri24_pd_few_cartesian_6x','sri24_pd_few_cartesian_7x','sri24_pd_few_cartesian_8x','sri24_pd_few_cartesian_9x','sri24_pd_few_gaussian_4x','sri24_pd_few_gaussian_5x','sri24_pd_few_gaussian_6x','sri24_pd_few_gaussian_7x','sri24_pd_few_gaussian_8x','sri24_pd_few_gaussian_9x','sri24_t2_few_cartesian_4x','sri24_t2_few_cartesian_5x','sri24_t2_few_cartesian_6x','sri24_t2_few_cartesian_7x','sri24_t2_few_cartesian_8x','sri24_t2_few_cartesian_9x','sri24_t2_few_gaussian_4x','sri24_t2_few_gaussian_5x','sri24_t2_few_gaussian_6x','sri24_t2_few_gaussian_7x','sri24_t2_few_gaussian_8x','sri24_t2_few_gaussian_9x'
#EVALUATE_TASK_STRINGS='mrbrain_ir_few_gaussian_4x','mrbrain_ir_few_gaussian_5x','mrbrain_ir_few_gaussian_8x','mrbrain_ir_few_cartesian_4x','mrbrain_ir_few_cartesian_5x','mrbrain_ir_few_cartesian_8x'

for DATASET_TYPE in 'sri24_t1_few' 'sri24_t2_few' 'sri24_pd_few'
    do
    for MASK_TYPE in 'cartesian' 'gaussian'
        do 
        for ACC_FACTOR in '4x' '5x' '6x' '7x' '8x' '9x'
            do 
            echo ${DATASET_TYPE}','${MASK_TYPE}','${ACC_FACTOR} 
            REPORT_PATH=${BASE_REPORT_PATH}'/report_'${DATASET_TYPE}'_'${MASK_TYPE}'_'${ACC_FACTOR}'.txt'
            echo ${REPORT_PATH}
            cat ${REPORT_PATH}
            echo "\n"
            done 
        done 
    done 

