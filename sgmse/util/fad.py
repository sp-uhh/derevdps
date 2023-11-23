from frechet_audio_distance import FrechetAudioDistance


def FAD(gt_dir, generated_dir, backbone="vggish", dtype="float32"):
    
    if backbone == "vggish":
        frechet = FrechetAudioDistance(
            model_name="vggish",
            sample_rate=16000,
            use_pca=False, 
            use_activation=False,
            verbose=False
        )
    elif backbone == "pann":
        frechet = FrechetAudioDistance(
            model_name="pann",
            sample_rate=16000,
            use_pca=False, 
            use_activation=False,
            verbose=False
        )
    elif backbone == "clap":
        frechet = FrechetAudioDistance(
            model_name="clap",
            sample_rate=48000,
            submodel_name="630k-audioset",  # for CLAP only
            verbose=False,
            enable_fusion=False,            # for CLAP only
        )

    return frechet.score(gt_dir, generated_dir, dtype="float32")
    

if __name__ == "__main__":

    gt_dir = "/data/lemercier/databases/wsj0_derev_with_rir/audio/tt/clean_sub"
    generated_dir = "/data/lemercier/databases/wsj0_derev_with_rir/audio/tt/noisy_sub"
    generated_dir = gt_dir

    print(FAD(gt_dir, generated_dir))