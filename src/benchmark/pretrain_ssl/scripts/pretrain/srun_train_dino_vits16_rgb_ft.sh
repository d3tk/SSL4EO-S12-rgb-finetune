#!/usr/bin/env bash
#SBATCH --job-name=ssl4eo_rgb_vits # A descriptive job name
#SBATCH --output=ssl4eo_rgb_vits_%j.out  # Output file; %j will be replaced with the job ID
#SBATCH --nodes=1   # Use a single node
#SBATCH --gres=gpu:4     # Request 2 GPUs on that node
#SBATCH --cluster=gpu    # Specify the GPU cluster
#SBATCH --partition=ls40s     # Use the ls40s partition (adjust if needed)
#SBATCH --time=24:00:00  # Walltime (days-HH:MM:SS); adjust as necessary
#SBATCH --mail-user=DTK28@pitt.edu # Your Pitt email for notifications
#SBATCH --mail-type=BEGIN,END,FAIL # Send email when job ends or fails

# load required modules
module purge
module load gcc/12.2.0
module load python/anaconda3.10-2022.10

# activate virtualenv or conda env if needed
source .venv/bin/activate

export WANDB_API_KEY="1d6d3552cc426c9580fad66684d5950088c5210d"

srun torchrun --nproc_per_node=4 pretrain_dino_rgb.py \
    --seed 42 \
    --arch vit_small \
    --patch_size 16 \
    --out_dim 65536 \
    --momentum_teacher 0.996 \
    --warmup_teacher_temp 0.04 \
    --teacher_temp 0.04 \
    --warmup_teacher_temp_epochs 0 \
    --weight_decay 0.04 \
    --weight_decay_end 0.4 \
    --clip_grad 3.0 \
    --batch_size_per_gpu 144 \
    --epochs 100 \
    --freeze_last_layer 1 \
    --lr 0.0005 \
    --warmup_epochs 10 \
    --min_lr 1e-6 \
    --optimizer adamw \
    --drop_path_rate 0.1 \
    --checkpoints_dir ~/checkpoints/ssl4eo/dino/vits/rgb \
    --saveckp_freq 20 \
    --num_workers 1 \
    --dist_url env:// \
    --data ~/datasets/ssl4eo/0k_251k_uint8_jpeg_tif \
    --bands RGB \
    --is_slurm_job \
    --normalize \
    --mode rgb \
    --dtype uint8 \
    --season augment  
    --strategy single   
    --in_size 224 \
    --wandb_project SSL4EO-RUNS \
    --wandb_entity msc-thesis-proj \
    --wandb_run_name ssl4eo_vits_rgb_finetune \
    --wandb_log --wandb_run_name ssl4eo_vits_rgb_finetune

# srun torchrun --nproc_per_node=1 pretrain_dino_rgb.py \
#     --seed 42 \
#     --arch vit_small \
#     --patch_size 16 \
#     --out_dim 65536 \
#     --momentum_teacher 0.996 \
#     --warmup_teacher_temp 0.04 \
#     --teacher_temp 0.04 \
#     --warmup_teacher_temp_epochs 0 \
#     --weight_decay 0.04 \
#     --weight_decay_end 0.4 \
#     --clip_grad 3.0 \
#     --batch_size_per_gpu 8 \
#     --epochs 100 \
#     --freeze_last_layer 1 \
#     --lr 0.0005 \
#     --warmup_epochs 10 \
#     --min_lr 1e-6 \
#     --optimizer adamw \
#     --drop_path_rate 0.1 \
#     --checkpoints_dir checkpoints/ssl4eo/dino/vits/rgb \
#     --saveckp_freq 20 \
#     --num_workers 1 \
#     --dist_url env:// \
#     --data /mnt/e/data/ssl4eo/0k_251k_uint8_jpeg_tif \
#     --bands RGB \
#     --is_slurm_job \
#     --normalize \
#     --mode rgb \
#     --dtype uint8 \
#     --season augment  
#     --strategy single   
#     --in_size 224 \
#     --wandb_project SSL4EO-RUNS \
#     --wandb_entity msc-thesis-proj \
#     --wandb_run_name ssl4eo_vits_rgb_finetune \
#     --wandb_log --wandb_run_name ssl4eo_vits_rgb_finetune