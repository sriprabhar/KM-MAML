MODEL='<model_name>'  
BASE_PATH='<base folder path>'

EXPERIMENT_PATH=${BASE_PATH}'/<path to where model folder is saved>'

CHECKPOINT=${EXPERIMENT_PATH}'/'${MODEL}'/best_model.pt'

RESULTS_DIR=${EXPERIMENT_PATH}'/'${MODEL}'/results' # path to where predicted images are stored

TENSORBOARD_SUMMARY_DIR=${EXPERIMENT_PATH}'/'${MODEL}'/'

USMASK_PATH=${BASE_PATH}'/usmasks/'

TEST_SUPPORT_BATCH_SIZE=20

DEVICE='cuda:0'

TEST_PATH=${BASE_PATH}

NUM_TEST_ADAPTATION_STEPS=30

TEST_TASK_STRINGS='mrbrain_t1_few_gaussian_4x','mrbrain_t1_few_gaussian_5x','mrbrain_t1_few_gaussian_8x','mrbrain_t1_few_cartesian_4x','mrbrain_t1_few_cartesian_5x','mrbrain_t1_few_cartesian_8x','mrbrain_flair_few_gaussian_4x','mrbrain_flair_few_gaussian_5x','mrbrain_flair_few_gaussian_8x','mrbrain_flair_few_cartesian_4x','mrbrain_flair_few_cartesian_5x','mrbrain_flair_few_cartesian_8x'

#TEST_TASK_STRINGS='mrbrain_t1_few_gaussian_4x','mrbrain_t1_few_gaussian_5x','mrbrain_t1_few_gaussian_8x','mrbrain_t1_few_cartesian_4x','mrbrain_t1_few_cartesian_5x','mrbrain_t1_few_cartesian_8x','mrbrain_flair_few_gaussian_4x','mrbrain_flair_few_gaussian_5x','mrbrain_flair_few_gaussian_8x','mrbrain_flair_few_cartesian_4x','mrbrain_flair_few_cartesian_5x','mrbrain_flair_few_cartesian_8x'

echo python test.py --checkpoint ${CHECKPOINT} --results_dir ${RESULTS_DIR} --tensorboard_summary_dir ${TENSORBOARD_SUMMARY_DIR} --usmask_path ${USMASK_PATH} --test_support_batch_size ${TEST_SUPPORT_BATCH_SIZE} --device ${DEVICE} --test_path ${TEST_PATH} --no_of_test_adaptation_steps ${NUM_TEST_ADAPTATION_STEPS} --test_task_strings ${TEST_TASK_STRINGS}

python test.py --checkpoint ${CHECKPOINT} --results_dir ${RESULTS_DIR} --tensorboard_summary_dir ${TENSORBOARD_SUMMARY_DIR} --usmask_path ${USMASK_PATH} --test_support_batch_size ${TEST_SUPPORT_BATCH_SIZE} --device ${DEVICE} --test_path ${TEST_PATH} --no_of_test_adaptation_steps ${NUM_TEST_ADAPTATION_STEPS} --test_task_strings ${TEST_TASK_STRINGS}










