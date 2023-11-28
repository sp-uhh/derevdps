
gt_dir=/data/lemercier/databases/wsj0_derev_with_rir/audio/tt/clean



python unconditional.py \
    --gt_dir $gt_dir \
    --generated_dir .test_eh \
    --ckpt .logs/epoch=106.ckpt \
    --sampler_type karras \
    --predictor euler-heun \
    --scheduler edm \
    --N 50 \
    --n 25