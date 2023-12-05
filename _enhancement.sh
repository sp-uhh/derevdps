#!/bin/bash

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
# n=2

# N=200
# scheduler="edm"
# # zeta=0.3
# zeta=7
# # zeta=1000
# r=0.4
# alpha=1
# beta=0.1
# pre="karras"
# sde="edm"

# # zeta_schedule="constant"
# zeta_schedule="div-sig"
# # zeta_schedule="saw-tooth-increase"

# sampler_type="karras"

# # predictor="euler-maruyama"
# # predictor="euler-heun"
# predictor="euler-heun-dps"

# corrector="none"
# # corrector="ald"

# # posterior="dps"
# posterior="none"

# ckpt_score="$home_dir/code/_public_repos/derevdps/.logs/sde=EDM_backbone=ncsnpp_data=wsj0_ch=1/version_5/checkpoints/epoch=146.ckpt"

# python3 enhancement.py \
#     --test_dir $test_dir --rir_dir $rir_dir \
#     --N $N --n $n --sampler_type $sampler_type --scheduler $scheduler \
#     --predictor $predictor \
#     --corrector $corrector --r $r \
#     --posterior $posterior --operator reverberation --zeta $zeta --zeta_schedule $zeta_schedule \
#     --ckpt $ckpt_score \
#     --enhanced_dir .exp/.posterior/prior=0_${sampler_type}_sde=${sde}_pre=${pre}_alpha=${alpha}_beta=${beta}_N=${N}_pred=${predictor}_corr=${corrector}_r=${r}_sched=${scheduler}_post=${posterior}_zeta=${zeta}_zsched=${zeta_schedule}











#####################
### Test RED-Diff ###
#####################

# test_dir="$data_dir/wsj0_derev_with_rir/audio/tt/noisy"
# clean_dir="$data_dir/wsj0_derev_with_rir/audio/tt/clean"
# rir_dir="$data_dir/wsj0_derev_with_rir/rir/tt"
test_dir="$data_dir/vctk_derev_with_rir/audio/tt/noisy"
clean_dir="$data_dir/vctk_derev_with_rir/audio/tt/clean"
rir_dir="$data_dir/vctk_derev_with_rir/rir/tt"
n=1

N=200
scheduler="edm"
# zeta=0.25
zeta_schedule="none"
r=0.4
alpha=1
beta=0.1
pre="karras"
sde="edm"

sampler_type="red-diff"
lr=1e-2
optimizer="adam"
# lr=1e-2
# lr=1e3
# optimizer="sgd"
# zeta=0.01
zeta=0.88
# zeta=0.75


predictor="none"
corrector="none"
posterior="none"

# ckpt_score="$home_dir/code/_public_repos/derevdps/.logs/sde=EDM_backbone=ncsnpp_data=wsj0_ch=1/version_5/checkpoints/epoch=146.ckpt"
ckpt_score="$home_dir/code/_public_repos/derevdps/.logs/sde=EDM_backbone=ncsnpp_data=wsj0_ch=1/version_5/checkpoints/epoch=146.ckpt"

python3 enhancement.py \
    --test_dir $test_dir --rir_dir $rir_dir \
    --N $N --n $n --sampler_type $sampler_type --scheduler $scheduler \
    --predictor $predictor \
    --corrector $corrector --r $r \
    --posterior $posterior --operator reverberation --zeta $zeta --zeta_schedule $zeta_schedule \
    --optimizer $optimizer --lr $lr \
    --ckpt $ckpt_score \
    --enhanced_dir .exp/.posterior/prior=0_${sampler_type}_sde=${sde}_pre=${pre}_alpha=${alpha}_beta=${beta}_N=${N}_zeta=${zeta}_optimizer=${optimizer}_lr=${lr}