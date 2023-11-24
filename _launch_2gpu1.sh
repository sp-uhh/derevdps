#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2  # When using DDP, one task/process will be launched for each GPU 
#SBATCH --cpus-per-task=8          # We have 64 total in spgpu2 and 32 in spgpu1, making it 8 cores per GPU process in both cases
#SBATCH --partition=all
#SBATCH --nodelist=spgpu1          # Or set it to spgpu1
#SBATCH --job-name=edm_wsj0_0.1_17
#SBATCH --output=.slurm/%x-%j.out    # Save to folder ./jobs, %x means the job name. You may need to create this folder
#SBATCH --error=.slurm/%x-%j.err
#SBATCH --time=4-00:00             # Limit job to 4 days
#SBATCH --mem=0                    # SLURM does not limit the memory usage, but it will block jobs from launching
#SBATCH --qos=wimi-compute
#SBATCH --gres=gpu:2        # Number of GPUs to allocate

# source .environment/bin/activate

pc=spgpu1

if [ "$pc" = sppc1 ]; then
    data_dir=/data/lemercier/databases
    home_dir=/export/home/lemercier
elif [ "$pc" = spgpu1 ]; then
    data_dir=/data/lemercier/databases
    home_dir=/data1/lemercier
elif [ "$pc" = spgpu2 ]; then
    data_dir=/data3/lemercier/databases
    home_dir=/export/home/lemercier
fi;

# # WSJ0 Song Scale Factor = 0.1
# base_dir="$data_dir/wsj0+chime3/audio"
# format="wsj0"
# srun -K1 -u python3 train.py \
#     --backbone ncsnpp \
#     --format  $format \
#     --base_dir $base_dir \
#     --testset_dir $data_dir/wsj0_derev_with_rir \
#     --batch_size 8 \
#     --gpus 2 \
#     --spec_abs_exponent 1. \
#     --spec_factor 0.1 \
#     --condition none \
#     --sde ve \
#     --preconditioning song \
#     --num_eval_files 10 \
#     --num_unconditional_files 5 \
#     --sigma_min 0.00001 \
#     --sigma_max 17 \
#     --resume_from_checkpoint $home_dir/code/_public_repos/derevdps/.logs/sde=VESDE_backbone=ncsnpp_data=wsj0_ch=1/version_2/checkpoints/last.ckpt



base_dir="$data_dir/wsj0+chime3/audio"
format="wsj0"
# WSJ0 EDM Scale Factor = 0.1
srun -K1 -u python3 train.py \
    --backbone ncsnpp \
    --format  $format \
    --base_dir $base_dir \
    --testset_dir $data_dir/wsj0_derev_with_rir \
    --batch_size 8 \
    --gpus 2 \
    --spec_abs_exponent 1. \
    --spec_factor 0.1 \
    --condition none \
    --sde edm \
    --preconditioning karras_eloi \
    --num_eval_files 10 \
    --num_unconditional_files 25 \
    --sigma_min 0.00001 \
    --sigma_max 17 \
    --sigma_data 0.15 \
    --resume_from_checkpoint /data1/lemercier/code/_public_repos/derevdps/.logs/sde=EDM_backbone=ncsnpp_data=wsj0_ch=1/version_6/checkpoints/last.ckpt


# base_dir="$data_dir/wsj0+chime3/audio"
# format="wsj0"
# # WSJ0 EDM Scale Factor = 1 SIGMA 170
# srun -K1 -u python3 train.py \
#     --backbone ncsnpp \
#     --format  $format \
#     --base_dir $base_dir \
#     --testset_dir $data_dir/wsj0_derev_with_rir \
#     --batch_size 8 \
#     --gpus 2 \
#     --spec_abs_exponent 1. \
#     --spec_factor 0.1 \
#     --condition none \
#     --sde edm \
#     --preconditioning karras_eloi \
#     --num_eval_files 10 \
#     --num_unconditional_files 25 \
#     --sigma_min 0.00001 \
#     --sigma_max 170 \
#     --sigma_data 1.5


# # WSJ0 Song Scale Factor = 0.1 SIGMA_MAX = 1.5
# base_dir="$data_dir/wsj0+chime3/audio"
# format="wsj0"
# srun -K1 -u python3 train.py \
#     --backbone ncsnpp \
#     --format  $format \
#     --base_dir $base_dir \
#     --testset_dir $data_dir/wsj0_derev_with_rir \
#     --batch_size 8 \
#     --gpus 2 \
#     --spec_abs_exponent 1. \
#     --spec_factor 0.1 \
#     --condition none \
#     --sde ve \
#     --preconditioning song \
#     --num_eval_files 10 \
#     --num_unconditional_files 25 \
#     --sigma_min 0.01 \
#     --sigma_max 1.5 