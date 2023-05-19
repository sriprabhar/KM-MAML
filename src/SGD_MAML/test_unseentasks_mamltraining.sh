MODEL='<model_name>'  
BASE_PATH='<base folder path>'

EXPERIMENT_PATH=${BASE_PATH}'/<path to where model folder is saved>'

CHECKPOINT=${EXPERIMENT_PATH}'/'${MODEL}'/best_model.pt'

RESULTS_DIR=${EXPERIMENT_PATH}'/'${MODEL}'/results_unseen_adapt_30' # path to where predicted images are stored


TENSORBOARD_SUMMARY_DIR=${EXPERIMENT_PATH}'/'${MODEL}'/'

USMASK_PATH=${BASE_PATH}'/usmasks/'

TEST_SUPPORT_BATCH_SIZE=5

DEVICE='cuda:0'

TEST_PATH=${BASE_PATH}

NUM_TEST_ADAPTATION_STEPS=10

#TEST_TASK_STRINGS='sri24pd_few_cartesian_4x','sri24pd_few_cartesian_5x','sri24pd_few_cartesian_6x','sri24pd_few_cartesian_7x','sri24pd_few_cartesian_8x','sri24pd_few_cartesian_9x','sri24pd_few_gaussian_4x','sri24pd_few_gaussian_5x','sri24pd_few_gaussian_6x','sri24pd_few_gaussian_7x','sri24pd_few_gaussian_8x','sri24pd_few_gaussian_9x','sri24t2_few_cartesian_4x','sri24t2_few_cartesian_5x','sri24t2_few_cartesian_6x','sri24t2_few_cartesian_7x','sri24t2_few_cartesian_8x','sri24t2_few_cartesian_9x','sri24pd_few_gaussian_4x','sri24pd_few_gaussian_5x','sri24pd_few_gaussian_6x','sri24pd_few_gaussian_7x','sri24pd_few_gaussian_8x','sri24pd_few_gaussian_9x'

TEST_TASK_STRINGS='sri24_pd_few_cartesian_4x','sri24_pd_few_cartesian_5x','sri24_pd_few_cartesian_6x','sri24_pd_few_cartesian_7x','sri24_pd_few_cartesian_8x','sri24_pd_few_cartesian_9x','sri24_pd_few_gaussian_4x','sri24_pd_few_gaussian_5x','sri24_pd_few_gaussian_6x','sri24_pd_few_gaussian_7x','sri24_pd_few_gaussian_8x','sri24_pd_few_gaussian_9x','sri24_t2_few_cartesian_4x','sri24_t2_few_cartesian_5x','sri24_t2_few_cartesian_6x','sri24_t2_few_cartesian_7x','sri24_t2_few_cartesian_8x','sri24_t2_few_cartesian_9x','sri24_t2_few_gaussian_4x','sri24_t2_few_gaussian_5x','sri24_t2_few_gaussian_6x','sri24_t2_few_gaussian_7x','sri24_t2_few_gaussian_8x','sri24_t2_few_gaussian_9x','sri24_t1_few_cartesian_4x','sri24_t1_few_cartesian_5x','sri24_t1_few_cartesian_6x','sri24_t1_few_cartesian_7x','sri24_t1_few_cartesian_8x','sri24_t1_few_cartesian_9x','sri24_t1_few_gaussian_4x','sri24_t1_few_gaussian_5x','sri24_t1_few_gaussian_6x','sri24_t1_few_gaussian_7x','sri24_t1_few_gaussian_8x','sri24_t1_few_gaussian_9x'
#TEST_TASK_STRINGS='mrbrain_ir_few_gaussian_4x','mrbrain_ir_few_gaussian_5x','mrbrain_ir_few_gaussian_8x','mrbrain_ir_few_cartesian_4x','mrbrain_ir_few_cartesian_5x','mrbrain_ir_few_cartesian_8x'

#TEST_TASK_STRINGS='mrbrain_t1_few_gaussian_4x','mrbrain_t1_few_gaussian_5x','mrbrain_t1_few_gaussian_8x','mrbrain_t1_few_cartesian_4x','mrbrain_t1_few_cartesian_5x','mrbrain_t1_few_cartesian_8x','mrbrain_flair_few_gaussian_4x','mrbrain_flair_few_gaussian_5x','mrbrain_flair_few_gaussian_8x','mrbrain_flair_few_cartesian_4x','mrbrain_flair_few_cartesian_5x','mrbrain_flair_few_cartesian_8x'

echo python test.py --checkpoint ${CHECKPOINT} --results_dir ${RESULTS_DIR} --tensorboard_summary_dir ${TENSORBOARD_SUMMARY_DIR} --usmask_path ${USMASK_PATH} --test_support_batch_size ${TEST_SUPPORT_BATCH_SIZE} --device ${DEVICE} --test_path ${TEST_PATH} --no_of_test_adaptation_steps ${NUM_TEST_ADAPTATION_STEPS} --test_task_strings ${TEST_TASK_STRINGS}

python test.py --checkpoint ${CHECKPOINT} --results_dir ${RESULTS_DIR} --tensorboard_summary_dir ${TENSORBOARD_SUMMARY_DIR} --usmask_path ${USMASK_PATH} --test_support_batch_size ${TEST_SUPPORT_BATCH_SIZE} --device ${DEVICE} --test_path ${TEST_PATH} --no_of_test_adaptation_steps ${NUM_TEST_ADAPTATION_STEPS} --test_task_strings ${TEST_TASK_STRINGS}










