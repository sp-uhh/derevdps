#!/usr/env/bin/python3
import os
from os.path import join
import numpy as np
import pyroomacoustics as pra
from abc import ABC, abstractmethod
import torch, torchaudio
from glob import glob

# SEED = 100
# np.random.seed(SEED)

# rev_params = {
# 	"t60_range": [0.4, 1.0],
# 	"dim_range": [5, 15, 5, 15, 2, 6],
# 	"min_distance_to_wall": 1.
# }

class RIRKernel(ABC):

    @abstractmethod
    def forward(self, **kernel_kwargs):
        pass

    @abstractmethod
    def sample_kernel_kwargs(self, **kernel_kwargs):
        pass

class SimulateRIRKernel(RIRKernel):

    def forward(self, **kernel_kwargs):

        self.sample_kernel_kwargs(**kernel_kwargs)
        k = self.simulate_kernel()
        return torch.from_numpy(k)
        
    def simulate_kernel(self, fs=16000):

        center_mic_position = np.array([ self.room_dim[0]//2, self.room_dim[1]//2, self.room_dim[2]//2 ])
        source_angle = np.random.uniform(0, 2*np.pi)
        source_position = np.array([ 
            center_mic_position[0] + np.cos(source_angle)*self.distance, 
            center_mic_position[1] + np.sin(source_angle)*self.distance,
            center_mic_position[2]
            ])
        mic_array_2d = pra.beamforming.circular_2D_array(center_mic_position[: -1], 1, phi0=0, radius=1.) # Compute microphone array
        mic_array = np.pad(mic_array_2d, ((0, 1), (0, 0)), mode="constant", constant_values=center_mic_position[-1])

        e_absorption, max_order = pra.inverse_sabine(self.t60, self.room_dim) #Compute absorption coeff
        reverberant_room = pra.ShoeBox(
            self.room_dim, fs=fs, materials=pra.Material(e_absorption), max_order=min(3, max_order), ray_tracing=True
        ) # Create room
        reverberant_room.set_ray_tracing()
        reverberant_room.add_microphone_array(mic_array) # Add microphone array

        # Generate reverberant room
        reverberant_room.add_source(source_position)
        reverberant_room.compute_rir()
        return reverberant_room.rir[0][0]

    def sample_kernel_kwargs(self, **kernel_kwargs):
        self.t60 = np.random.uniform(kernel_kwargs["t60_range"][0], kernel_kwargs["t60_range"][1])
        self.room_dim = [np.random.uniform(kernel_kwargs["dim_range"][2*i], kernel_kwargs["dim_range"][2*i+1]) for i in range(3)]
        max_distance_admissible = np.min(np.array([ 
            self.room_dim[0] // 2 - kernel_kwargs["min_distance_to_wall"], 
            self.room_dim[0] // 1 - kernel_kwargs["min_distance_to_wall"], 
            kernel_kwargs["distance_range"][1] 
            ]))
        self.distance = np.random.uniform(kernel_kwargs["distance_range"][0], max_distance_admissible)

class RealRIRKernel(RIRKernel):

    def forward(self, **kernel_kwargs):

        self.sample_kernel_kwargs(**kernel_kwargs)
        k, _ = torchaudio.load(self.kernel_path)
        return k.squeeze(0)

    def sample_kernel_kwargs(self, **kernel_kwargs):
        if "i" in kernel_kwargs.keys():
            self.kernel_path = join(kernel_kwargs["rir_path"], sorted(os.listdir(kernel_kwargs["rir_path"]))[kernel_kwargs["i"]])
        else:
            self.kernel_path = np.random.choice(glob(join(kernel_kwargs["rir_path"], "*.wav")))

class InitializeRIRKernel(RIRKernel):
    
    def forward(self, sr=16000, **kernel_kwargs):

        noise = torch.randn(int(self.L*sr))
        decay_env = torch.exp(-3*torch.linspace(0, self.L, int(self.L*sr)) / (np.log(10)*self.T60) )
        rir = noise * decay_env
        return rir / rir.abs().max()
        
    def sample_kernel_kwargs(self, **kernel_kwargs):
        self.L = 2
        self.T60 = 0.1
