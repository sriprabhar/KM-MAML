MODEL='<model_name>'  
BASE_PATH='<base folder path>'

# Note that the dataset folder needs to organised in terms of tasks. For MRI reconstruction, each task folder is in the hierarchy of datasset type / mask type/ acceleration factor/ 
# inside the acceleration factor the train and validation folders must be present
# for arranging the datasets based on tasks, please refer to the README in https://github.com/sriprabhar/MAC-ReconNet
# for maml training, each dataset has a train and validation partition. Train is further divided into train_support and train_query. Validation partition is further divided into validate_support and validate query

# during meta-training - the train_support and train_query are used for the inner and outper loop iterations

# for adaptation to few gradient steps - the valid_support partition is used
# for evaluation /  testing the valid_query partition is used

# for training on source domains ('mrbrain_t1' 'mrbrain_flair' 'ixi_pd' 'ixi_t2') and evaluating on target unseen domains ('sri24_t1', 'sri24_t2' and 'sri24_pd') again these unseen datasets are partitioned as validation_support and validation_query datasets

DATASET_TYPE='<dataset strings seperated by commas>' #'mrbrain_t1_few','mrbrain_flair_few','ixi_pd_few','ixi_t2_few'
MASK_TYPE='<mask pattern types, seperated by commas>' #'cartesian','gaussian'
ACC_FACTORS='4x','5x','8x'

TRAIN_TASK_STRINGS='mrbrain_t1_few_gaussian_4x','mrbrain_t1_few_gaussian_5x','mrbrain_t1_few_gaussian_8x','mrbrain_t1_few_cartesian_4x','mrbrain_t1_few_cartesian_5x','mrbrain_t1_few_cartesian_8x','mrbrain_flair_few_gaussian_4x','mrbrain_flair_few_gaussian_5x','mrbrain_flair_few_gaussian_8x','mrbrain_flair_few_cartesian_4x','mrbrain_flair_few_cartesian_5x','mrbrain_flair_few_cartesian_8x','ixi_pd_few_gaussian_4x','ixi_pd_few_gaussian_5x','ixi_pd_few_gaussian_8x','ixi_pd_few_cartesian_4x','ixi_pd_few_cartesian_5x','ixi_pd_few_cartesian_8x','ixi_t2_few_gaussian_4x','ixi_t2_few_gaussian_5x','ixi_t2_few_gaussian_8x','ixi_t2_few_cartesian_4x','ixi_t2_few_cartesian_5x','ixi_t2_few_cartesian_8x'


TRAIN_SUPPORT_BATCH_SIZE=10
VAL_SUPPORT_BATCH_SIZE=10

TRAIN_QUERY_BATCH_SIZE=10
VAL_QUERY_BATCH_SIZE=10

TRAIN_TASK_BATCH_SIZE=3
VAL_TASK_BATCH_SIZE=6

TRAIN_NUM_ADAPTATION_STEPS=1
VAL_NUM_ADAPTATION_STEPS=1

NUM_EPOCHS=600
DEVICE='cuda:0'


EXP_DIR=${BASE_PATH}'/experiments/maml/'${MODEL}
TRAIN_PATH=${BASE_PATH}

#VALIDATION_PATH=${BASE_PATH}'/datasets/'

USMASK_PATH=${BASE_PATH}

# run a unet model in unsupervised mode (like an autoencoder) using the under-sampled input in order to get the embedding vector from the bottleneck layer
# this embedding vector is used to drive the kernel modulation network

DISENTANGLE_MODEL_PATH='/<path to model file of the autoencoder trained in unsupervised model using the under-sampled images>/best_model.pt'

echo python train.py --train_task_batch_size ${TRAIN_TASK_BATCH_SIZE} --val_task_batch_size ${VAL_TASK_BATCH_SIZE} --task_strings ${TRAIN_TASK_STRINGS} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --train_path ${TRAIN_PATH} --dataset_type ${DATASET_TYPE} --usmask_path ${USMASK_PATH} --acceleration_factor ${ACC_FACTORS} --mask_type ${MASK_TYPE} --train_support_batch_size ${TRAIN_SUPPORT_BATCH_SIZE} --train_query_batch_size ${TRAIN_QUERY_BATCH_SIZE} --val_support_batch_size ${VAL_SUPPORT_BATCH_SIZE} --val_query_batch_size ${VAL_QUERY_BATCH_SIZE} --no_of_train_adaptation_steps ${TRAIN_NUM_ADAPTATION_STEPS} --no_of_val_adaptation_steps ${VAL_NUM_ADAPTATION_STEPS}

python train.py --train_task_batch_size ${TRAIN_TASK_BATCH_SIZE} --val_task_batch_size ${VAL_TASK_BATCH_SIZE} --task_strings ${TRAIN_TASK_STRINGS} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --train_path ${TRAIN_PATH} --dataset_type ${DATASET_TYPE} --usmask_path ${USMASK_PATH} --acceleration_factor ${ACC_FACTORS} --mask_type ${MASK_TYPE} --train_support_batch_size ${TRAIN_SUPPORT_BATCH_SIZE} --train_query_batch_size ${TRAIN_QUERY_BATCH_SIZE} --val_support_batch_size ${VAL_SUPPORT_BATCH_SIZE} --val_query_batch_size ${VAL_QUERY_BATCH_SIZE} --no_of_train_adaptation_steps ${TRAIN_NUM_ADAPTATION_STEPS} --no_of_val_adaptation_steps ${VAL_NUM_ADAPTATION_STEPS} --disentangle-model-path ${DISENTANGLE_MODEL_PATH}
