export TASK_NAME=sst2
export DATA_DIR=/home/fourteen/workspace/dataset/nlp/glue_data/SST-2
# export TASK_NAME=cola
# export TASK_NAME=mnli
# export TASK_NAME=sst2
# export TASK_NAME=stsb
# export TASK_NAME=qqp
# export TASK_NAME=qnli
# export TASK_NAME=rte

# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 WORLD_SIZE=8 python -m torch.distributed.launch --nproc_per_node 8 --master_port 42476 run_glue.py \
CUDA_VISIBLE_DEVICES=0 python run_glue_recover.py \
  --model_name_or_path textattack/bert-base-uncased-SST-2 \
  --data_dir $DATA_DIR \
  --task_name $TASK_NAME \
  --per_device_train_batch_size 1 \