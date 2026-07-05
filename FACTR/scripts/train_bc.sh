
#!/bin/bash

# cuda id
CUDA_DEVICE_ID=0

# task configuration, setup under cfg/task/
task_config=single_franka

# path to dataset buffer
buffer_path=$(pwd)/processed_data/test/buf.pkl

# curriculum parameters
space_config=pixel # pixel, latent
scheduler_config=linear # no, const, linear, step, exp, cos
operator_config=blur  # blur, downsample 
start_scale=5
stop_scale=0

# pretrained visual features
feature_path=$(pwd)/visual_features/vit_base/SOUP_1M_DH.pth

# wandb
wandb_entity=YOUR_WANDB

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE_ID python factr/train_bc_policy.py \
agent.features.restore_path=$feature_path \
buffer_path=$buffer_path \
task=$task_config \
curriculum.space=$space_config \
curriculum.operator=$operator_config \
curriculum.scheduler=$scheduler_config \
curriculum.start_scale=$start_scale \
curriculum.stop_scale=$stop_scale \
wandb.entity=wandb_entity