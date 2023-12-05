import abc

import torch, torchaudio
import numpy as np

from sgmse.util.registry import Registry

### Actual operations ###

class RIRTimeConv(torch.nn.Module):
    def __init__(self, time_kernel_size=10):
        """
        Create a placeholder for a convolution in the time domain
        """
        super().__init__()
        self.time_kernel_size = time_kernel_size
        self.seq = torch.nn.Sequential(
            torch.nn.ConstantPad1d((self.time_kernel_size-1, 0), 0.),
            torch.nn.Conv1d(1, 1, self.time_kernel_size, stride=1, padding=0, bias=False)
        )

    def forward(self, x):
        return self.seq(x)

    def update_weights(self, k, kernel_size, **ignored_kwargs):
        k = k[: kernel_size]
        padding = self.time_kernel_size - k.size(-1)
        if padding > 0:
            k = torch.nn.functional.pad(k, (0, padding))
        else:
            k = k[..., : self.time_kernel_size]
        k = torch.flip(k, dims=(-1,)).unsqueeze(0).unsqueeze(1)
        self.k = k
        for _, f in self.named_parameters():
            f.data = k

    def get_kernel(self):
        return self.k



OperatorRegistry = Registry("Operator")

class Operator(abc.ABC):

    @abc.abstractmethod
    def forward(self, x, **kwargs):
        pass

class LinearOperator(Operator):
    
    @abc.abstractmethod
    def forward(self, x, **kwargs):
        # calculate A * X
        pass

    @abc.abstractmethod
    def transpose(self, x, **kwargs):
        # calculate A^T * X
        pass
    
    def ortho_project(self, x, **kwargs):
        # calculate (I - A^T * A)X
        return x - self.transpose(self.forward(x, **kwargs), **kwargs)

    def project(self, x, y, **kwargs):
        # calculate (I - A^T * A)Y - AX
        return self.ortho_project(y, **kwargs) - self.forward(x, **kwargs)



@OperatorRegistry.register('noise')
class DenoiseOperator(LinearOperator):
    def __init__(self, *args, **kwargs):
        pass

    def forward(self, x):
        return x

    def transpose(self, x):
        return x
    
    def ortho_project(self, x):
        return x

    def project(self, x):
        return x


@OperatorRegistry.register('reverberation')
class ReverberationOperator(LinearOperator):
    def __init__(self, kernel_size, stft=False, **stft_kwargs):
        self.kernel_size = kernel_size
        self.stft = stft
        if self.stft:
            raise NotImplementedError
            self.conv = RIRTimeFreqConv(time_kernel_size=kernel_size)
        else:
            self.conv = RIRTimeConv(time_kernel_size=kernel_size)
        self.stft_kwargs = stft_kwargs

    def load_weights(self, kernel):
        self.conv.update_weights(kernel, kernel_size=self.kernel_size, **self.stft_kwargs) #crop to kernel_size

    def forward(self, x, **kwargs):
        # A^T * A 
        return self.conv(x)

    def transpose(self, x, **kwargs):
        return x
    
    # def pinv(self, x, kernel):
    #     nfft_if = min(2*int(16000*1.6), x.size(-1))
    #     X = torch.stft(x, **self.stft_kwargs)
    #     RTF = torch.fft.rfft(kernel, n=nfft_if).unsqueeze(-1)
    #     regularized_iRTF = torch.conj(RTF) / (torch.square(torch.abs(RTF)) + 1e-2) 
    #     S = X * regularized_iRTF
    #     s = torch.istft(S, **self.stft_kwargs)
    #     return s
    
    def pinv(self, x, kernel):
        nfft_if = min(2*int(16000*1.6), x.size(-1))        
        local_istft_kwargs = {
            "n_fft": nfft_if,
            "hop_length": nfft_if//2,
            "window": torch.hann_window(nfft_if).to(x.device),
            "center": True
        }
        local_stft_kwargs = {
            **local_istft_kwargs,
            "return_complex": True
        }

        X = torch.stft(x, **local_stft_kwargs)
        RTF = torch.fft.rfft(kernel, n=nfft_if).unsqueeze(-1)
        regularized_iRTF = torch.conj(RTF) / (torch.square(torch.abs(RTF)) + 1e-2) 
        S = X * regularized_iRTF
        s = torch.istft(S, **local_istft_kwargs)
        return s
    
    def get_kernel(self):
        return self.conv.get_kernel()

@OperatorRegistry.register('none')
class NoOperator(LinearOperator):

    def forward(self, x, **kwargs):
        return x