MODEL='<model_name>'  
BASE_PATH='<base folder path>'

EXPERIMENT_PATH=${BASE_PATH}'/<path to where model folder is saved>'

BASE_TARGET_PATH=${BASE_PATH}'/datasets/'


BASE_PREDICTIONS_PATH=${EXPERIMENT_PATH}'/'${MODEL}'/results_unseen_adapt_30/' # path to where predictions are stored

REPORT_PATH=${EXPERIMENT_PATH}'/'${MODEL}'/results_unseen_adapt_30/'

#EVALUATE_TASK_STRINGS='mrbrain_t1_few_gaussian_4x','mrbrain_t1_few_gaussian_5x','mrbrain_t1_few_gaussian_8x','mrbrain_t1_few_cartesian_4x','mrbrain_t1_few_cartesian_5x','mrbrain_t1_few_cartesian_8x','mrbrain_flair_few_gaussian_4x','mrbrain_flair_few_gaussian_5x','mrbrain_flair_few_gaussian_8x','mrbrain_flair_few_cartesian_4x','mrbrain_flair_few_cartesian_5x','mrbrain_flair_few_cartesian_8x'

#EVALUATE_TASK_STRINGS='mrbrain_ir_few_gaussian_4x','mrbrain_ir_few_gaussian_5x','mrbrain_ir_few_gaussian_8x','mrbrain_ir_few_cartesian_4x','mrbrain_ir_few_cartesian_5x','mrbrain_ir_few_cartesian_8x'

EVALUATE_TASK_STRINGS='sri24_pd_few_cartesian_4x','sri24_pd_few_cartesian_5x','sri24_pd_few_cartesian_6x','sri24_pd_few_cartesian_7x','sri24_pd_few_cartesian_8x','sri24_pd_few_cartesian_9x','sri24_pd_few_gaussian_4x','sri24_pd_few_gaussian_5x','sri24_pd_few_gaussian_6x','sri24_pd_few_gaussian_7x','sri24_pd_few_gaussian_8x','sri24_pd_few_gaussian_9x','sri24_t2_few_cartesian_4x','sri24_t2_few_cartesian_5x','sri24_t2_few_cartesian_6x','sri24_t2_few_cartesian_7x','sri24_t2_few_cartesian_8x','sri24_t2_few_cartesian_9x','sri24_t2_few_gaussian_4x','sri24_t2_few_gaussian_5x','sri24_t2_few_gaussian_6x','sri24_t2_few_gaussian_7x','sri24_t2_few_gaussian_8x','sri24_t2_few_gaussian_9x','sri24_t1_few_cartesian_4x','sri24_t1_few_cartesian_5x','sri24_t1_few_cartesian_6x','sri24_t1_few_cartesian_7x','sri24_t1_few_cartesian_8x','sri24_t1_few_cartesian_9x','sri24_t1_few_gaussian_4x','sri24_t1_few_gaussian_5x','sri24_t1_few_gaussian_6x','sri24_t1_few_gaussian_7x','sri24_t1_few_gaussian_8x','sri24_t1_few_gaussian_9x'

python evaluate_slicewise.py --base-target-path ${BASE_TARGET_PATH} --base-predictions-path ${BASE_PREDICTIONS_PATH} --report-path ${REPORT_PATH} --evaluate_task_strings ${EVALUATE_TASK_STRINGS}
 
