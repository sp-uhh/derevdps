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

# source .environment/bin/activate

# test_dir="/data/lemercier/databases/wsj0_derev_with_rir/audio/tt/noisy"
# clean_dir="/data/lemercier/databases/wsj0_derev_with_rir/audio/tt/clean"
# n=2

# N=50
# alpha="1.0"
# beta="0.1"
# scheduler="ve-song"
# # scheduler="linear"
# # zeta=50 ### Attention: since we added the dt multiplicator in the posterior now to make it more clean, we need to multiply our dewfault zeta values by N=50
# # zeta=2500
# zeta=60 ### Compounded effects of dt multiplication and normguide
# r=0.4

# pre="song"
# sde="ve"

# zeta_schedule="saw-tooth-increase"
# sampler_type="song"
# predictor="euler-maruyama"
# corrector="ald"
# posterior="dps"

# ckpt_score="/export/home/lemercier/code/score_derev/.logs/official_checkpoints/checkpoints_derevdps_waspaa2023/score-only_unconditional_sde=ve_pre=song_alpha=1.0_beta=0.1_sigma=0.5_epoch=204.ckpt"

# python3 enhancement.py \
#     --test_dir $test_dir \
#     --N $N --n $n --sampler_type $sampler_type --scheduler $scheduler \
#     --predictor $predictor \
#     --corrector $corrector --r $r \
#     --posterior $posterior --operator reverberation --zeta $zeta --zeta_schedule $zeta_schedule \
#     --ckpt $ckpt_score \
#     --enhanced_dir ./results \
#     # --no_probability_flow
#     # --enhanced_dir .exp/.posterior/prior=0_${sampler_type}_sde=${sde}_pre=${pre}_alpha=${alpha}_beta=${beta}_N=${N}_pred=${predictor}_corr=${corrector}_r=${r}_sched=${scheduler}_post=${posterior}_zeta=${zeta}_zsched=${zeta_schedule}



###############
## UNIFY XP ###
###############

# test_dir="$data_dir/wsj0_derev_with_rir/audio/tt/noisy"
# clean_dir="$data_dir/wsj0_derev_with_rir/audio/tt/clean"
# rir_dir="$data_dir/wsj0_derev_with_rir/rir/tt"
# ckpt_score="$home_dir/code/_public_repos/derevdps/.logs/sde=EDM_backbone=ncsnpp_data=wsj0_ch=1/version_5/checkpoints/epoch=146.ckpt"

test_dir="$data_dir/vctk_derev_with_rir/audio/tt/noisy"
clean_dir="$data_dir/vctk_derev_with_rir/audio/tt/clean"
rir_dir="$data_dir/vctk_derev_with_rir/rir/tt"
ckpt_score="/export/home/lemercier/code/_public_repos/derevdps/.logs/sde=EDM_backbone=ncsnpp_data=vctk_pretarget/version_1/checkpoints/epoch=253.ckpt"

n=2

N=200
scheduler="edm"
# zeta=0.3
zeta=7
# zeta=1000
r=0.4
alpha=1
beta=0.1
pre="karras"
sde="edm"

# zeta_schedule="constant"
zeta_schedule="div-sig"
# zeta_schedule="saw-tooth-increase"

sampler_type="karras"

# predictor="euler-maruyama"
# predictor="euler-heun"
predictor="euler-heun-dps"

corrector="none"
# corrector="ald"

# posterior="dps"
posterior="none"


python3 enhancement.py \
    --test_dir $test_dir --rir_dir $rir_dir \
    --N $N --n $n --sampler_type $sampler_type --scheduler $scheduler \
    --predictor $predictor \
    --corrector $corrector --r $r \
    --posterior $posterior --operator reverberation --zeta $zeta --zeta_schedule $zeta_schedule \
    --ckpt $ckpt_score \
    --enhanced_dir .exp/.posterior/prior=0_${sampler_type}_sde=${sde}_pre=${pre}_alpha=${alpha}_beta=${beta}_N=${N}_pred=${predictor}_corr=${corrector}_r=${r}_sched=${scheduler}_post=${posterior}_zeta=${zeta}_zsched=${zeta_schedule}











#####################
### Test RED-Diff ###
#####################

# test_dir="$data_dir/wsj0_derev_with_rir/audio/tt/noisy"
# clean_dir="$data_dir/wsj0_derev_with_rir/audio/tt/clean"
# rir_dir="$data_dir/wsj0_derev_with_rir/rir/tt"
# ckpt_score="$home_dir/code/_public_repos/derevdps/.logs/sde=EDM_backbone=ncsnpp_data=wsj0_ch=1/version_5/checkpoints/epoch=146.ckpt"

test_dir="$data_dir/vctk_derev_with_rir/audio/tt/noisy"
clean_dir="$data_dir/vctk_derev_with_rir/audio/tt/clean"
rir_dir="$data_dir/vctk_derev_with_rir/rir/tt"
ckpt_score="/export/home/lemercier/code/_public_repos/derevdps/.logs/sde=EDM_backbone=ncsnpp_data=vctk_pretarget/version_1/checkpoints/epoch=253.ckpt"

n=3

N=200
scheduler="edm"
zeta_schedule="none"
alpha=1
beta=1
pre="karras"
sde="edm"

sampler_type="red-diff"
# lr=5e-1
lr=1e-1
# lr=1e-2
optimizer="adam"
# zeta=0.88
zeta=0.5
# zeta=0.3
# zeta=0.1
zeta_schedule="constant"

predictor="none"
corrector="none"
posterior="none"


python3 enhancement.py \
    --test_dir $test_dir --rir_dir $rir_dir \
    --N $N --n $n --sampler_type $sampler_type --scheduler $scheduler \
    --predictor $predictor \
    --corrector $corrector --r $r \
    --churn $churn \
    --posterior $posterior --operator reverberation --zeta $zeta --zeta_schedule $zeta_schedule \
    --optimizer $optimizer --lr $lr \
    --ckpt $ckpt_score \
    --enhanced_dir .exp/.posterior/prior=0_${sampler_type}_sde=${sde}_pre=${pre}_alpha=${alpha}_beta=${beta}_N=${N}_zeta=${zeta}_optimizer=${optimizer}_lr=${lr}
