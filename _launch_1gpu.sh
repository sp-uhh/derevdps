#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2  # When using DDP, one task/process will be launched for each GPU 
#SBATCH --cpus-per-task=8          # We have 64 total in spgpu2 and 32 in spgpu1, making it 8 cores per GPU process in both cases
#SBATCH --partition=all
#SBATCH --nodelist=spgpu2          # Or set it to spgpu1
#SBATCH --job-name=song_wsj0
#SBATCH --output=.slurm/%x-%j.out    # Save to folder ./jobs, %x means the job name. You may need to create this folder
#SBATCH --error=.slurm/%x-%j.err
#SBATCH --time=4-00:00             # Limit job to 4 days
#SBATCH --mem=0                    # SLURM does not limit the memory usage, but it will block jobs from launching
#SBATCH --qos=wimi-compute
#SBATCH --gres=gpu:2        # Number of GPUs to allocate

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
