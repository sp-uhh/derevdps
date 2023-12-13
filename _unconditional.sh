
# gt_dir=/data/lemercier/databases/wsj0_derev_with_rir/audio/tt/clean

# python unconditional.py \
#     --gt_dir $gt_dir \
#     --generated_dir .test_eh \
#     --ckpt .logs/epoch=106.ckpt \
#     --sampler_type karras \
#     --predictor euler-heun \
#     --scheduler edm \
#     --N 50 \
#     --n 25

# With cheating Song

# gt_dir=/data/lemercier/databases/vctk_derev_with_rir/audio/tt/clean
    
# python unconditional.py \
#     --gt_dir $gt_dir \
#     --generated_dir .test_em_song_vctk \
#     --ckpt .logs/epoch=150.ckpt \
#     --sampler_type song \
#     --predictor euler-maruyama \
#     --corrector none \
#     --scheduler ve \
#     --N 500 \
#     --n 2

gt_dir=/data/lemercier/databases/wsj0_derev_with_rir/audio/tt/clean
    
python unconditional.py \
    --gt_dir $gt_dir \
    --generated_dir .test_em_song_wsj0 \
    --ckpt .logs/epoch=259.ckpt \
    --sampler_type song \
    --predictor euler-maruyama \
    --corrector none \
    --scheduler ve \
    --N 100 \
    --n 2