#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1  # When using DDP, one task/process will be launched for each GPU 
#SBATCH --cpus-per-task=8          # We have 64 total in spgpu2 and 32 in spgpu1, making it 8 cores per GPU process in both cases
#SBATCH --partition=all
#SBATCH --nodelist=spgpu2          # Or set it to spgpu1
#SBATCH --job-name=song_wsj0
#SBATCH --output=.slurm/%x-%j.out    # Save to folder ./jobs, %x means the job name. You may need to create this folder
#SBATCH --error=.slurm/%x-%j.err
#SBATCH --time=4-00:00             # Limit job to 4 days
#SBATCH --mem=0                    # SLURM does not limit the memory usage, but it will block jobs from launching
#SBATCH --qos=wimi-compute
#SBATCH --gres=gpu:1        # Number of GPUs to allocate

source .environment/bin/activate


base_dir="/data/lemercier/databases/wsj0+chime3/audio"
format="wsj0"

# WSJ0 Song Scale Factor = 0.1
srun -K1 -u python3 train.py \
    --backbone ncsnpp \
    --format  $format \
    --base_dir $base_dir \
    --testset_dir /data/lemercier/databases/wsj0_derev_with_rir \
    --batch_size 16 \
    --gpus 2 \
    --spec_abs_exponent 1. \
    --spec_factor 0.1 \
    --condition none \
    --sde ve \
    --preconditioning song \
    --num_eval_files 10 \
    --num_unconditional_files 5 \
    --sigma_min 0.00001 \
    --sigma_max 17

# # WSJ0 Song Scale Factor = 1.0
# srun -K1 -u python3 train.py \
#     --mode score-only \
#     --backbone_score ncsnpp \
#     --format  $format \
#     --base_dir $base_dir \
#     --batch_size 16 \
#     --gpus 2 \
#     --spec_abs_exponent 1. \
#     --spec_factor 1. \
#     --condition none \
#     --sde ve \
#     --preconditioning song \
#     --num_eval_files 10 \
#     --num_unconditional_files 5 \
#     --sigma_min 0.00001 \
#     --sigma_max 170

# WSJ0 EDM Scale Factor = 0.1
srun -K1 -u python3 train.py \
    --mode score-only \
    --backbone_score ncsnpp \
    --format  $format \
    --base_dir $base_dir \
    --batch_size 16 \
    --gpus 2 \
    --spec_abs_exponent 1. \
    --spec_factor 0.1 \
    --condition none \
    --sde edm \
    --preconditioning karras_eloi \
    --num_eval_files 10 \
    --num_unconditional_files 5 \
    --sigma_min 0.00001 \
    --sigma_max 17 \
    --sigma_data 0.15

# # WSJ0 EDM Scale Factor = 1.
# srun -K1 -u python3 train.py \
#     --mode score-only \
#     --backbone_score ncsnpp \
#     --format  $format \
#     --base_dir $base_dir \
#     --batch_size 16 \
#     --gpus 2 \
#     --spec_abs_exponent 1. \
#     --spec_factor 1 \
#     --condition none \
#     --sde edm \
#     --preconditioning karras_eloi \
#     --num_eval_files 10 \
#     --num_unconditional_files 5 \
#     --sigma_min 0.00001 \
#     --sigma_max 170 \
#     --sigma_data 1.5





# VCTK Song Scale Factor = 0.1
srun -K1 -u python3 train.py \
    --mode score-only \
    --backbone_score ncsnpp \
    --format  $format \
    --base_dir $base_dir \
    --batch_size 16 \
    --gpus 2 \
    --spec_abs_exponent 1. \
    --spec_factor 0.1 \
    --condition none \
    --sde ve \
    --preconditioning song \
    --num_eval_files 10 \
    --num_unconditional_files 5 \
    --sigma_min 0.00001 \
    --sigma_max 15

# VCTK Song Scale Factor = 1.0
srun -K1 -u python3 train.py \
    --mode score-only \
    --backbone_score ncsnpp \
    --format  $format \
    --base_dir $base_dir \
    --batch_size 16 \
    --gpus 2 \
    --spec_abs_exponent 1. \
    --spec_factor 1. \
    --condition none \
    --sde ve \
    --preconditioning song \
    --num_eval_files 10 \
    --num_unconditional_files 5 \
    --sigma_min 0.00001 \
    --sigma_max 150

# VCTK EDM Scale Factor = 0.1
srun -K1 -u python3 train.py \
    --mode score-only \
    --backbone_score ncsnpp \
    --format  $format \
    --base_dir $base_dir \
    --batch_size 16 \
    --gpus 2 \
    --spec_abs_exponent 1. \
    --spec_factor 0.1 \
    --condition none \
    --sde edm \
    --preconditioning karras_eloi \
    --num_eval_files 10 \
    --num_unconditional_files 5 \
    --sigma_min 0.00001 \
    --sigma_max 15 \
    --sigma_data 0.17

# VCTK EDM Scale Factor = 1. Comparison with Eloi
srun -K1 -u python3 train.py \
    --mode score-only \
    --backbone_score ncsnpp \
    --format  $format \
    --base_dir $base_dir \
    --batch_size 16 \
    --gpus 2 \
    --spec_abs_exponent 1. \
    --spec_factor 1 \
    --condition none \
    --sde edm \
    --preconditioning karras_eloi \
    --num_eval_files 10 \
    --num_unconditional_files 5 \
    --sigma_min 0.00001 \
    --sigma_max 150 \
    --sigma_data 1.7




































base_dir="/data/lemercier/databases/wsj0+chime3/audio"
format="wsj0"
python3 train.py \
    --backbone ncsnpp \
    --format  $format \
    --base_dir $base_dir \
    --testset_dir /data/lemercier/databases/wsj0_derev_with_rir \
    --batch_size 2 \
    --gpus 1 \
    --spec_abs_exponent 1. \
    --spec_factor 0.1 \
    --condition none \
    --sde ve \
    --preconditioning song \
<<<<<<< HEAD
    --num_eval_files 10 \
    --num_unconditional_files 25 \
    --sigma_min 0.01 \
    --sigma_max 1.5 \
=======
    --num_eval_files 2 \
    --num_unconditional_files 2 \
    --sigma_min 0.00001 \
    --sigma_max 17 \
>>>>>>> 890e35d0052810b49604829bdd6068acbbf1a9b2
    --limit_train_batches 25 \
    --limit_val_batches 10