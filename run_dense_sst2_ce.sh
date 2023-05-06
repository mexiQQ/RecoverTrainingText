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
CUDA_VISIBLE_DEVICES=0 python run_glue.py \
  --model_name_or_path textattack/bert-base-uncased-SST-2 \
  --data_dir $DATA_DIR \
  --task_name $TASK_NAME \
  --max_length 128 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 64 \
  --learning_rate 5e-5 \
  --eval_step 200 \
  --print_step 1 \
  --seed 42 \
  --num_train_epochs 1 \
  --lr_scheduler_type constant_with_warmup \
  --output_dir runs/sst2 \
  # --do_eval