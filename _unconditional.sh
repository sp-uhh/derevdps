
gt_dir=/data/lemercier/databases/vctk_derev_with_rir/audio/tt/clean

python unconditional.py \
    --gt_dir $gt_dir \
    --generated_dir .exp/edm_karrassampler_eh \
    --ckpt .logs/sde=EDM_backbone=ncsnpp_data=vctk_pretarget/version_1/checkpoints/epoch=253.ckpt \
    --sampler_type karras \
    --predictor euler-heun \
    --scheduler edm \
    --N 100 \
    --n 5