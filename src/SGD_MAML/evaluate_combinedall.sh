MODEL='<model_name>'  
BASE_PATH='<base folder path>'


EXPERIMENT_PATH=${BASE_PATH}'/experiments/maml'

BASE_TARGET_PATH=${BASE_PATH}'/datasets/'


BASE_PREDICTIONS_PATH=${EXPERIMENT_PATH}'/'${MODEL}'/results/'

REPORT_PATH=${EXPERIMENT_PATH}'/'${MODEL}'/'

EVALUATE_TASK_STRINGS='mrbrain_t1_few_gaussian_4x','mrbrain_t1_few_gaussian_5x','mrbrain_t1_few_gaussian_8x','mrbrain_t1_few_cartesian_4x','mrbrain_t1_few_cartesian_5x','mrbrain_t1_few_cartesian_8x','mrbrain_flair_few_gaussian_4x','mrbrain_flair_few_gaussian_5x','mrbrain_flair_few_gaussian_8x','mrbrain_flair_few_cartesian_4x','mrbrain_flair_few_cartesian_5x','mrbrain_flair_few_cartesian_8x'

python evaluate_slicewise.py --base-target-path ${BASE_TARGET_PATH} --base-predictions-path ${BASE_PREDICTIONS_PATH} --report-path ${REPORT_PATH} --evaluate_task_strings ${EVALUATE_TASK_STRINGS}
 
