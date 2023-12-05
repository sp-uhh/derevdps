#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1  # When using DDP, one task/process will be launched for each GPU 
#SBATCH --cpus-per-task=8          # We have 64 total in spgpu2 and 32 in spgpu1, making it 8 cores per GPU process in both cases
#SBATCH --partition=all
#SBATCH --nodelist=spgpu2          # Or set it to spgpu1
#SBATCH --job-name=song_vctk_1_15_pt
#SBATCH --output=.slurm/%x-%j.out    # Save to folder ./jobs, %x means the job name. You may need to create this folder
#SBATCH --error=.slurm/%x-%j.err
#SBATCH --time=4-00:00             # Limit job to 4 days
#SBATCH --mem=0                    # SLURM does not limit the memory usage, but it will block jobs from launching
#SBATCH --qos=wimi-compute
#SBATCH --gres=gpu:1        # Number of GPUs to allocate

# source .environment/bin/activate

pc=spgpu2

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

# # VCTK EDM Scale Factor = 1 (comparison Eloi)
# base_dir=$data_dir/vctk_56spk/audio
# format="vctk"
# srun -K1 -u python3 train.py \
#     --backbone ncsnpp \
#     --format  $format \
#     --base_dir $base_dir \
#     --testset_dir $data_dir/vctk_derev_with_rir \
#     --batch_size 16 \
#     --gpus 1 \
#     --spec_abs_exponent 1. \
#     --spec_factor 1 \
#     --condition none \
#     --sde edm \
#     --preconditioning karras \
#     --num_eval_files 10 \
#     --num_unconditional_files 25 \
#     --sigma_min 1e-5 \
#     --sigma_max 150 \
#     --sigma_data 1.7


# VCTK Song Scale Factor = 1 Sigma_max = 5 
base_dir=$data_dir/vctk_56spk/audio
format="vctk"
srun -K1 -u python3 train.py \
    --backbone ncsnpp \
    --format  $format \
    --base_dir $base_dir \
    --testset_dir $data_dir/vctk_derev_with_rir \
    --batch_size 16 \
    --gpus 1 \
    --spec_abs_exponent 1. \
    --spec_factor 1 \
    --condition none \
    --sde ve \
    --preconditioning song \
    --num_eval_files 10 \
    --num_unconditional_files 25 \
    --sigma_min 1e-5 \
    --sigma_max 15 \
    --t_eps 1e-3

# # VCTK Song Scale Factor = 1 Sigma_max = 1.5 
# base_dir=$data_dir/vctk_56spk/audio
# format="vctk"
# srun -K1 -u python3 train.py \
#     --backbone ncsnpp \
#     --format  $format \
#     --base_dir $base_dir \
#     --testset_dir $data_dir/vctk_derev_with_rir \
#     --batch_size 16 \
#     --gpus 1 \
#     --spec_abs_exponent 1. \
#     --spec_factor 1 \
#     --condition none \
#     --sde ve \
#     --preconditioning song \
#     --num_eval_files 10 \
#     --num_unconditional_files 25 \
#     --sigma_min 1e-5 \
#     --sigma_max 1.5 \
#     --t_eps 1e-3

















# python3 train.py \
#     --backbone ncsnpp \
#     --format  $format \
#     --base_dir $base_dir \
#     --testset_dir $data_dir/vctk_derev_with_rir \
#     --batch_size 2 \
#     --gpus 2, \
#     --spec_abs_exponent 1. \
#     --spec_factor 1 \
#     --condition none \
#     --sde edm \
#     --preconditioning karras \
#     --num_eval_files 10 \
#     --num_unconditional_files 25 \
#     --sigma_min 1e-5 \
#     --sigma_max 150 \
#     --sigma_data 1.7 \
#     --limit_train_batches 25 \
#     --limit_val_batches 10