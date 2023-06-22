import torch
from torchaudio import load
import matplotlib.pyplot as plt
import numpy as np
import os
import soundfile as sf
import glob

# Plotting settings
EPS_graphics = 1e-10
n_fft = 512
hop_length = 128

stft_kwargs = {"n_fft": n_fft, "hop_length": hop_length, "window": torch.hann_window(n_fft), "center": True, "return_complex": True}

def visualize_example(mix, estimate, target, idx_sample=0, epoch=0, name="", sample_rate=16000, hop_len=128, return_fig=False):
	"""Visualize training targets and estimates of the Neural Network
	Args:
		- mix: Tensor [F, T]
		- estimates/targets: Tensor [F, T]
	"""

	if isinstance(mix, torch.Tensor):
		mix = torch.abs(mix).detach().cpu()
		estimate = torch.abs(estimate).detach().cpu()
		target = torch.abs(target).detach().cpu()

	vmin, vmax = -60, 0
	fac = .5

	fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(24, 8))

	freqs = sample_rate/(2*mix.size(-2)) * torch.arange(mix.size(-2))
	frames = hop_len/sample_rate * torch.arange(mix.size(-1))

	ax = axes.flat[0]
	im = ax.pcolormesh(frames, freqs, 20*np.log10(fac*mix + EPS_graphics), vmin=vmin, vmax=vmax, shading="auto", cmap="magma")
	ax.set_xlabel('Time [s]')
	ax.set_ylabel('Frequency [Hz]')
	ax.set_title('Mixed Speech')

	freqs = sample_rate/(2*estimate.size(-2)) * torch.arange(estimate.size(-2))
	frames = hop_len/sample_rate * torch.arange(estimate.size(-1))

	ax = axes.flat[1]
	ax.pcolormesh(frames, freqs, 20*np.log10(fac*estimate + EPS_graphics), vmin=vmin, vmax=vmax, shading="auto", cmap="magma")
	ax.set_xlabel('Time [s]')
	ax.set_ylabel('Frequency [Hz]')
	ax.set_title('Anechoic estimate')

	freqs = sample_rate/(2*target.size(-2)) * torch.arange(target.size(-2))
	frames = hop_len/sample_rate * torch.arange(target.size(-1))

	ax = axes.flat[2]
	ax.pcolormesh(frames, freqs, 20*np.log10(fac*target + EPS_graphics), vmin=vmin, vmax=vmax, shading="auto", cmap="magma")
	ax.set_xlabel('Time [s]')
	ax.set_ylabel('Frequency [Hz]')
	ax.set_title('Anechoic target')

	fig.subplots_adjust(right=0.87)
	cbar_ax = fig.add_axes([0.9, 0.25, 0.005, 0.5])
	fig.colorbar(im, cax=cbar_ax)

	return fig