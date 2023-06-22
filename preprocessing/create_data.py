#!/usr/env/bin/python3

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from os.path import join
import numpy as np
import soundfile as sf
import glob
import argparse
import time
import json
from tqdm import tqdm
import shutil
import scipy.signal as ss
import io 
import scipy.io.wavfile 
import pyroomacoustics as pra

from utils import obtain_noise_file

SEED = 100
np.random.seed(SEED)

derev_params = {
    "t60_range": [0.4, 1.0],
    "dim_range": [5, 15, 5, 15, 2, 6],
    "min_distance_to_wall": 1.
}

# ROOT = "" ## put your root directory here
ROOT = "/data/lemercier/databases"
assert ROOT != "", "You need to have a root databases directory"

parser = argparse.ArgumentParser()

parser.add_argument('--speech', type=str, choices=["vctk", "wsj0", "dns", "timit"], default="wsj0", help='Clean speech')
parser.add_argument('--sr', type=int, default=16000)
parser.add_argument('--splits', type=str, default="cv,tr,tt", help='Split folders of the dataset')
parser.add_argument('--save-rir', type=bool, default=True)
parser.add_argument('--dummy', action="store_true", help='Number of samples')

args = parser.parse_args()
splits = args.splits.strip().split(",")
dic_splits = {"cv": "valid", "tt": "test", "tr": "train"}

params = vars(args)
params = {**derev_params, **params}

output_dir = join(ROOT, args.speech + "_" + args.task)
if args.save_rir:
    output_dir += "_with_rir"

if args.speech == "wsj0":
    dic_split = {"cv": "si_dt_05", "tr": "si_tr_s", "tt": "si_et_05"}
    speech_lists = {split:glob.glob(f"{ROOT}/WSJ0/wsj0/{dic_split[split]}/**/*.wav") for split in splits}
elif args.speech == "vctk":
    speakers = sorted(os.listdir(f"{ROOT}/VCTK-Corpus/wav48/"))
    speakers.remove("p280")
    speakers.remove("p315")
    ranges = {"tr": [0, 99], "cv": [97, 99], "tt": [99, 107]}
    speech_lists  = {split:[] for split in splits}
    for split in splits:
        for spk_idx in range(*ranges[split]):
            speech_lists[split] += glob.glob(f"{ROOT}/VCTK-Corpus/wav48/{speakers[spk_idx]}/*.wav")
elif args.speech == "timit":
    ranges = {"tr": [1, 7], "cv": [7, 8], "tt": [1, 8]}
    speech_lists  = {split:[] for split in splits}
    transcription_lists  = {split:[] for split in splits}
    for split in splits:
        splt_dr = "train" if split in ["cv", "tr"] else "test"
        for dr_idx in range(*ranges[split]):
            speech_lists[split] += sorted(glob.glob(f"{ROOT}/TIMIT/timit/{splt_dr}/dr{dr_idx}/**/*.wav"))
            transcription_lists[split] += sorted(glob.glob(f"{ROOT}/TIMIT/timit/{splt_dr}/dr{dr_idx}/**/*.txt"))

if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)
log = open(join(output_dir, "log_stats.txt"), "w")
log.write("Parameters \n ========== \n")
for key, param in params.items():
    log.write(key + " : " + str(param) + "\n")

for i_split, split in enumerate(splits):

    print("Processing split n° {}: {}...".format(i_split+1, split))
    
    clean_output_dir = join(output_dir, "audio", split, "clean")
    noisy_output_dir = join(output_dir, "audio", split, "noisy")
    os.makedirs(clean_output_dir, exist_ok=True)
    os.makedirs(noisy_output_dir, exist_ok=True)
    if "derev" in args.task and args.save_rir:
        rir_dir = join(output_dir, "rir", split)
        os.makedirs(rir_dir, exist_ok=True)
    if args.speech == "timit":
        transcription_output_dir = join(output_dir, "transcriptions", split)
        os.makedirs(transcription_output_dir, exist_ok=True)

    speech_list = speech_lists[split]
    speech_dir = None
    real_nb_samples = 5 if args.dummy else len(speech_list)

    for i_sample in tqdm(range(real_nb_samples)):

        speech_basename = os.path.basename(speech_list[i_sample])
        speech, sr = sf.read(speech_list[i_sample])
        assert sr == args.sr, "Obtained an unexpected Sampling rate"
        original_scale = np.max(np.abs(speech))

        lossy_speech = speech.copy()

        t60 = np.random.uniform(params["t60_range"][0], params["t60_range"][1]) #sample t60
        room_dim = np.array([ np.random.uniform(params["dim_range"][2*n], params["dim_range"][2*n+1]) for n in range(3) ]) #sample Dimensions
        center_mic_position = np.array([ np.random.uniform(params["min_distance_to_wall"], room_dim[n] - params["min_distance_to_wall"]) for n in range(3) ]) #sample microphone position
        source_position = np.array([ np.random.uniform(params["min_distance_to_wall"], room_dim[n] - params["min_distance_to_wall"]) for n in range(3) ]) #sample source position
        distance_source = 1/np.sqrt(center_mic_position.ndim)*np.linalg.norm(center_mic_position - source_position)
        mic_array_2d = pra.beamforming.circular_2D_array(center_mic_position[: -1], 1, phi0=0, radius=1.) # Compute microphone array
        mic_array = np.pad(mic_array_2d, ((0, 1), (0, 0)), mode="constant", constant_values=center_mic_position[-1])

        ### Reverberant Room
        e_absorption, max_order = pra.inverse_sabine(t60, room_dim) #Compute absorption coeff
        reverberant_room = pra.ShoeBox(
            room_dim, fs=16000, materials=pra.Material(e_absorption), max_order=min(3, max_order), ray_tracing=True
        ) # Create room
        reverberant_room.set_ray_tracing()
        reverberant_room.add_microphone_array(mic_array) # Add microphone array

        # Generate reverberant room
        reverberant_room.add_source(source_position, signal=lossy_speech)
        reverberant_room.compute_rir()
        reverberant_room.simulate()
        t60_real = np.mean(reverberant_room.measure_rt60()).squeeze()
        lossy_speech = np.squeeze(np.array(reverberant_room.mic_array.signals))

        #compute target
        e_absorption_dry = 0.99
        dry_room = pra.ShoeBox(
            room_dim, fs=16000, materials=pra.Material(e_absorption_dry), max_order=0
        ) # Create room
        dry_room.add_microphone_array(mic_array) # Add microphone array

        # Generate dry room
        dry_room.add_source(source_position, signal=speech) 
        dry_room.compute_rir()
        dry_room.simulate()
        t60_real_dry = np.mean(dry_room.measure_rt60()).squeeze()
        speech = np.squeeze(np.array(dry_room.mic_array.signals))
        noise_floor_snr = 50
        noise_floor_power = 1/speech.shape[0]*np.sum(speech**2)*np.power(10,-noise_floor_snr/10)
        noise_floor_signal = np.random.rand(int(.5*args.sr)) * np.sqrt(noise_floor_power)
        speech = np.concatenate([ speech, noise_floor_signal ])
        
        min_length = min(lossy_speech.shape[0], speech.shape[0])
        lossy_speech, speech = lossy_speech[: min_length], speech[: min_length]

        filename =  speech_basename[: -4] + f"_{i_sample}"
        filename += f"_t60={t60_real:.2f}.wav"

        ### Export
        sf.write(join(clean_output_dir, filename), speech, args.sr)
        sf.write(join(noisy_output_dir, filename), lossy_speech, args.sr)
        if args.speech == "timit":
            shutil.copy(transcription_lists[split][i_sample], join(transcription_output_dir, filename[: -4] + ".txt"))
        if args.save_rir:
            sf.write(join(rir_dir, filename), reverberant_room.rir[0][0], args.sr)