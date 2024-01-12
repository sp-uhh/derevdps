#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1  # When using DDP, one task/process will be launched for each GPU 
#SBATCH --cpus-per-task=8          # We have 64 total in spgpu2 and 32 in spgpu1, making it 8 cores per GPU process in both cases
#SBATCH --partition=all
#SBATCH --nodelist=spgpu2          # Or set it to spgpu1
#SBATCH --job-name=eval
#SBATCH --output=.slurm/%x-%j.out    # Save to folder ./jobs, %x means the job name. You may need to create this folder
#SBATCH --error=.slurm/%x-%j.err
#SBATCH --time=4-00:00             # Limit job to 4 days
#SBATCH --mem=0                    # SLURM does not limit the memory usage, but it will block jobs from launching
#SBATCH --qos=wimi-compute
#SBATCH --gres=gpu:1        # Number of GPUs to allocate

# source .environment/bin/activate

pc=sppc1

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

###############
## DPS ###
###############

test_dir="$data_dir/vctk_derev_with_rir/audio/tt/noisy"
rir_dir="$data_dir/vctk_derev_with_rir/rir/tt"
ckpt_score="/export/home/lemercier/code/_public_repos/derevdps/.logs/sde=EDM_backbone=ncsnpp_data=vctk_pretarget/version_1/checkpoints/epoch=253.ckpt"

n=2

N=200
scheduler="edm"
zeta=7
r=0.4
alpha=1
beta=0.1
pre="karras"
sde="edm"

zeta_schedule="div-sig"
sampler_type="karras"
predictor="euler-heun-dps"
corrector="none"
posterior="none"


python3 enhancement.py \
    --test_dir $test_dir --rir_dir $rir_dir \
    --N $N --n $n --sampler_type $sampler_type --scheduler $scheduler \
    --predictor $predictor \
    --corrector $corrector --r $r \
    --posterior $posterior --operator reverberation --zeta $zeta --zeta_schedule $zeta_schedule \
    --ckpt $ckpt_score \
    --enhanced_dir .exp/.posterior/prior=0_${sampler_type}_sde=${sde}_pre=${pre}_alpha=${alpha}_beta=${beta}_N=${N}_pred=${predictor}_corr=${corrector}_r=${r}_sched=${scheduler}_post=${posterior}_zeta=${zeta}_zsched=${zeta_schedule}