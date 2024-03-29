import torch
from torchaudio import load
import matplotlib.pyplot as plt
import numpy as np
import os
import soundfile as sf
import glob
# import plotly.express as px
# import pandas as pd

# Plotting settings
EPS_graphics = 1e-10
n_fft = 512
hop_length = 128

stft_kwargs = {"n_fft": n_fft, "hop_length": hop_length, "window": torch.hann_window(n_fft), "center": True, "return_complex": True}

def visualize_example(mix, estimate, target, spec_path=None, name="", sample_rate=16000, hop_len=128, return_fig=False):
    """Visualize training targets and estimates of the Neural Network
    Args:
        - mix: Tensor [F, T]
        - estimates/targets: Tensor [F, T]
    """

    if isinstance(mix, torch.Tensor):
        mix = mix.detach().cpu()
        estimate = estimate.detach().cpu()
        target = target.detach().cpu()
    elif isinstance(estimate, str):
        mix = np.squeeze(sf.read(mix)[0])
        estimate = np.squeeze(sf.read(estimate)[0])
        target = np.squeeze(sf.read(target)[0])
        xmax = 6.
        mix = torch.stft(torch.from_numpy(mix[..., : int(xmax*sample_rate)]), **stft_kwargs)
        estimate = torch.stft(torch.from_numpy(estimate[..., : int(xmax*sample_rate)]), **stft_kwargs)
        target = torch.stft(torch.from_numpy(target[..., : int(xmax*sample_rate)]), **stft_kwargs)

    vmin, vmax = -60, 0
    scale = max(mix.abs().max(), estimate.abs().max())

    mix = .9 * mix / scale
    estimate = .9 * estimate / scale
    target = .9 * target / scale

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(24, 8))

    freqs = sample_rate/(2*mix.size(-2)) * torch.arange(mix.size(-2))
    frames = hop_len/sample_rate * torch.arange(mix.size(-1))

    ax = axes.flat[0]
    im = ax.pcolormesh(frames, freqs, 20*np.log10(mix.abs() + EPS_graphics), vmin=vmin, vmax=vmax, shading="auto", cmap="magma")
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Frequency [Hz]')
    ax.set_title(f'Mixed Speech [Scale={scale:.2f}]')

    freqs = sample_rate/(2*estimate.size(-2)) * torch.arange(estimate.size(-2))
    frames = hop_len/sample_rate * torch.arange(estimate.size(-1))

    ax = axes.flat[1]
    ax.pcolormesh(frames, freqs, 20*np.log10(estimate.abs() + EPS_graphics), vmin=vmin, vmax=vmax, shading="auto", cmap="magma")
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Frequency [Hz]')
    ax.set_title(f'Estimate [Scale={scale:.2f}]')

    freqs = sample_rate/(2*target.size(-2)) * torch.arange(target.size(-2))
    frames = hop_len/sample_rate * torch.arange(target.size(-1))

    ax = axes.flat[2]
    ax.pcolormesh(frames, freqs, 20*np.log10(target.abs() + EPS_graphics), vmin=vmin, vmax=vmax, shading="auto", cmap="magma")
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Frequency [Hz]')
    ax.set_title(f'Clean [Scale={scale:.2f}]')

    fig.subplots_adjust(right=0.87)
    cbar_ax = fig.add_axes([0.9, 0.25, 0.005, 0.5])
    fig.colorbar(im, cax=cbar_ax)

    if return_fig:
        plt.close()
        return fig
    else:
        assert spec_path is not None
        plt.savefig(os.path.join(spec_path, name + ".png"), dpi=300, bbox_inches="tight")
        plt.close()



def visualize_one(estimate, spec_path=None, name="", sample_rate=16000, hop_len=128, raw=True, return_fig=False):
    """Visualize training targets and estimates of the Neural Network
    Args:
        - mix: Tensor [F, T]
        - estimates/targets: Tensor [F, T]
    """

    if isinstance(estimate, torch.Tensor):
        estimate = torch.abs(estimate).squeeze().detach().cpu()
    elif isinstance(estimate, str):
        estimate = np.squeeze(sf.read(estimate)[0])
        xmax = 6.
        estimate = estimate[..., : int(xmax*sample_rate)]
        estimate = torch.stft(torch.from_numpy(estimate), **stft_kwargs)

    vmin, vmax = -60, 0
    scale = estimate.abs().max()
    estimate = .9 * estimate / scale

    freqs = sample_rate/(2*estimate.size(-2)) * torch.arange(estimate.size(-2))
    frames = hop_len/sample_rate * torch.arange(estimate.size(-1))

    fig = plt.figure(figsize=(8, 8))
    im = plt.pcolormesh(frames, freqs, 20*np.log10(estimate.abs() + EPS_graphics), vmin=vmin, vmax=vmax, shading="auto", cmap="magma")

    if raw:
        plt.yticks([])
        plt.tick_params(left="off")
        plt.xticks([])
        plt.tick_params(bottom="off")
    else:
        plt.xlabel('Time [s]')
        plt.ylabel('Frequency [Hz]')
        plt.title(f'Estimate [Scale={scale:.2f}]')
        cbar_ax = fig.add_axes([0.93, 0.25, 0.03, 0.4])
        fig.colorbar(im, cax=cbar_ax)

    if return_fig:
        plt.close()
        return fig
    else:
        assert spec_path is not None
        plt.savefig(os.path.join(spec_path, name + ".png"), dpi=300, bbox_inches="tight")
        plt.close()


def plot_loss_by_sigma(log_loss, log_x=True, freescale=False):
        """
        log_loss: list of lists, each list is [t, loss_mean, loss_var]
        """
        #separate the list into 3 lists
        t_bins=[i[0] for i in log_loss]
        loss_mean=[i[1] for i in log_loss]
        loss_std=[i[2] for i in log_loss]

        fig = plt.figure(figsize=(4, 4))
        plt.bar(t_bins, loss_mean)
        plt.close()

        # df=pd.DataFrame.from_dict(
        #         {"t": t_bins, "loss": loss_mean, "std": loss_std
        #         }
        #         )
        # if freescale==True:
        #     fig= error_line('bar', data_frame=df, x="t", y="loss", error_y="std", log_x=log_x,  markers=True)
        # else:
        #     fig= error_line('bar', data_frame=df, x="t", y="loss", error_y="std", log_x=log_x,  markers=True, range_y=[0, 2])
    
        return fig

def error_line(error_y_mode='band', **kwargs):
    """Extension of `plotly.express.line` to use error bands."""
    ERROR_MODES = {'bar','band','bars','bands',None}
    if error_y_mode not in ERROR_MODES:
        raise ValueError(f"'error_y_mode' must be one of {ERROR_MODES}, received {repr(error_y_mode)}.")
    if error_y_mode in {'bar','bars',None}:
        fig = px.line(**kwargs)
    elif error_y_mode in {'band','bands'}:
        if 'error_y' not in kwargs:
            raise ValueError(f"If you provide argument 'error_y_mode' you must also provide 'error_y'.")
        figure_with_error_bars = px.line(**kwargs)
        fig = px.line(**{arg: val for arg,val in kwargs.items() if arg != 'error_y'})
        for data in figure_with_error_bars.data:
            x = list(data['x'])
            y_upper = list(data['y'] + data['error_y']['array'])
            y_lower = list(data['y'] - data['error_y']['array'] if data['error_y']['arrayminus'] is None else data['y'] - data['error_y']['arrayminus'])
            color = f"rgba({tuple(int(data['line']['color'].lstrip('#')[i:i+2], 16) for i in (0, 2, 4))},.3)".replace('((','(').replace('),',',').replace(' ','')
            fig.add_trace(
                go.Scatter(
                    x = x+x[::-1],
                    y = y_upper+y_lower[::-1],
                    fill = 'toself',
                    fillcolor = color,
                    line = dict(
                        color = 'rgba(255,255,255,0)'
                    ),
                    hoverinfo = "skip",
                    showlegend = False,
                    legendgroup = data['legendgroup'],
                    xaxis = data['xaxis'],
                    yaxis = data['yaxis'],
                )
            )
        # Reorder data as said here: https://stackoverflow.com/a/66854398/8849755
        reordered_data = []
        for i in range(int(len(fig.data)/2)):
            reordered_data.append(fig.data[i+int(len(fig.data)/2)])
            reordered_data.append(fig.data[i])
        fig.data = tuple(reordered_data)
    return fig


if __name__ == "__main__":

    visualize_example(*3*["/data3/lemercier/databases/wsj0_derev_with_rir/audio/tt/clean/444c0204_327_t60=0.50.wav"], return_fig=False, spec_path=".", name="test")
    # visualize_one("/data3/lemercier/databases/wsj0_derev_with_rir/audio/tt/clean/444c0204_327_t60=0.50.wav", raw=False, return_fig=False, spec_path=".", name="test")