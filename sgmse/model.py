from math import ceil
import warnings

import matplotlib.pyplot as plt

import torch
import pytorch_lightning as pl
from torch_ema import ExponentialMovingAverage
import wandb
import time
import os
import numpy as np
import torchaudio

from sgmse import sampling
from sgmse.sdes import SDERegistry, VESDE, EDM
from sgmse.backbones import BackboneRegistry
from sgmse.util.inference import evaluate_model
from sgmse.util.graphics import visualize_example, visualize_one, plot_loss_by_sigma
from sgmse.util.other import pad_spec, pad_time, si_sdr_torch
from sgmse.util.train_utils import SigmaLossLogger
from sgmse.sampling.schedulers import VESongScheduler, EDMScheduler

VIS_EPOCHS = 5

torch.autograd.set_detect_anomaly(True)

class ScoreModel(pl.LightningModule):
    def __init__(self,
        backbone: str = "ncsnpp", sde: str = "vesde", preconditioning = "song",
        lr: float = 1e-4, ema_decay: float = 0.999,
        t_eps: float = 3e-2, transform: str = 'none', nolog: bool = False,
        num_eval_files: int = 10, loss_type: str = 'mse', data_module_cls = None, 
        condition: str = "none",
        num_samples_unconditional: int = 1,
        **kwargs
    ):
        """
        Create a new ScoreModel.

        Args:
            backbone: The underlying backbone DNN that serves as a score-based model.
                Must have an output dimensionality equal to the input dimensionality.
            sde: The SDE to use for the diffusion.
            lr: The learning rate of the optimizer. (1e-4 by default).
            ema_decay: The decay constant of the parameter EMA (0.999 by default).
            t_eps: The minimum time to practically run for to avoid issues very close to zero (1e-5 by default).
            reduce_mean: If `True`, average the loss across data dimensions.
                Otherwise sum the loss across data dimensions.
        """
        # print(kwargs)
        super().__init__()
        # Initialize Backbone DNN
        dnn_cls = BackboneRegistry.get_by_name(backbone)
        chan_multiplier = 1 if ("return_time" in kwargs.keys() and kwargs["return_time"]) else 2 
        kwargs.update(input_channels=1*chan_multiplier)

        self.dnn = dnn_cls(**kwargs)
        # Initialize SDE
        sde_cls = SDERegistry.get_by_name(sde)
        self.sde = sde_cls(**kwargs)
        # Store hyperparams and save them
        self.preconditioning = preconditioning
        self.lr = lr
        self.ema_decay = ema_decay
        self.ema = ExponentialMovingAverage(self.parameters(), decay=self.ema_decay)
        self._error_loading_ema = False
        self.t_eps = t_eps
        self.loss_type = loss_type
        self.num_eval_files = num_eval_files
        self.num_samples_unconditional = num_samples_unconditional

        self.save_hyperparameters(ignore=['nolog'])
        self.data_module = data_module_cls(**kwargs)
        self.condition = condition
        self._reduce_op = lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

        if self.preconditioning == "karras":
            self.p_mean = kwargs["p_mean"]
            self.p_std = kwargs["p_std"]
            self.sigma_data = kwargs["sigma_data"]

        self.nolog = nolog
        # Just for logging loss versus sigma
        t_bins = np.linspace(0, self.sde.T, 20)
        sigma_bins = self.sde.scheduler.continuous_step(t_bins)
        self.loss_logger_diff = SigmaLossLogger(sigma_bins)

    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--lr", type=float, default=1e-4, help="The learning rate")
        parser.add_argument("--ema_decay", type=float, default=0.999, help="The parameter EMA decay constant (0.999 by default)")
        parser.add_argument("--t_eps", type=float, default=0.03, help="The minimum time (3e-2 by default)")
        parser.add_argument("--num_eval_files", type=int, default=10, help="Number of files for speech enhancement performance evaluation during training.")
        parser.add_argument("--loss_type", type=str, default="mse", choices=("mse", "mae", "gaussian_entropy", "kristina", "sisdr", "time_mse"), help="The type of loss function to use.")
        parser.add_argument("--condition", default="noisy", choices=["noisy", "none"])
        parser.add_argument("--preconditioning", default="song", choices=["song", "karras", "karras_eloi"])

        parser.add_argument("--sigma_data", type=float, default=1.7)
        parser.add_argument("--p_mean", type=float, default=-1.2)
        parser.add_argument("--p_std", type=float, default=1.2)
        parser.add_argument("--num_samples_unconditional", type=int, default=4, help="Number of generated unconditional samples during evaluation.")

        return parser

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def optimizer_step(self, *args, **kwargs):
        # Method overridden so that the EMA params are updated after each optimizer step
        super().optimizer_step(*args, **kwargs)
        self.ema.update(self.parameters())

    # on_load_checkpoint / on_save_checkpoint needed for EMA storing/loading
    def on_load_checkpoint(self, checkpoint):
        ema = checkpoint.get('ema', None)
        if ema is not None:
            self.ema.load_state_dict(checkpoint['ema'])
        else:
            self._error_loading_ema = True
            warnings.warn("EMA state_dict not found in checkpoint!")

    def on_save_checkpoint(self, checkpoint):
        checkpoint['ema'] = self.ema.state_dict()

    def train(self, mode, no_ema=False):
        res = super().train(mode)  # call the standard `train` method with the given mode
        if not self._error_loading_ema:
            if mode == False and not no_ema:
                # eval
                self.ema.store(self.parameters())        # store current params in EMA
                self.ema.copy_to(self.parameters())      # copy EMA parameters over current params for evaluation
            else:
                # train
                if self.ema.collected_params is not None:
                    self.ema.restore(self.parameters())  # restore the EMA weights (if stored)
        return res

    def eval(self, no_ema=False):
        return self.train(False, no_ema=no_ema)

    def _loss(self, err, sigma, err_time=None, err_mag=None):
        if self.loss_type == 'mse':
            losses = torch.square(err.abs())
            losses = self.preconditioning_loss(losses, sigma)
            loss = torch.mean(0.5*torch.sum(losses.reshape(losses.shape[0], -1), dim=-1))

        elif self.loss_type == 'mae':
            losses = err.abs()
            losses = self.preconditioning_loss(losses, sigma)
            loss = torch.mean(0.5*torch.sum(losses.reshape(losses.shape[0], -1), dim=-1))

        return loss

    def preconditioning_input(self, dnn_input, t):
        if self.preconditioning == "song":
            scale = 1.
        if self.preconditioning == "karras" or self.preconditioning == "karras_eloi":
            sigma = self.sde._std(t).squeeze()
            scale = 1/torch.sqrt( self.sigma_data**2 + sigma**2)
            if scale.ndim and scale.ndim < dnn_input.ndim:
                scale = scale.view(scale.size(0), *(1,)*(dnn_input.ndim - scale.ndim))

        return scale * dnn_input

    def preconditioning_noise(self, t):
        if self.preconditioning == "song":
            if not t.ndim:
                t = t.unsqueeze(0)
            c_noise = t
        if self.preconditioning == "song_sigma":
            sigma = self.sde._std(t).squeeze()
            c_noise = sigma.unsqueeze(0)
        if self.preconditioning == "karras" or self.preconditioning == "karras_eloi":
            sigma = self.sde._std(t).squeeze()
            if not sigma.ndim:
                sigma = sigma.unsqueeze(0)
            c_noise = sigma **.25

        return c_noise

    def preconditioning_output(self, dnn_output, t):
        if self.preconditioning == "song":
            sigma = self.sde._std(t).squeeze()
            scale = sigma
        if self.preconditioning == "karras" or self.preconditioning == "karras_eloi":
            sigma = self.sde._std(t).squeeze()
            scale = sigma * self.sigma_data / torch.sqrt( self.sigma_data**2 + sigma**2)
        if scale.ndim and scale.ndim < dnn_output.ndim:
            scale = scale.view(scale.size(0), *(1,)*(dnn_output.ndim - scale.ndim))

        return scale * dnn_output

    def preconditioning_skip(self, x, t):
        if self.preconditioning == "song":
            scale = 1.
        if self.preconditioning == "karras" or self.preconditioning == "karras_eloi":
            sigma = self.sde._std(t).squeeze()
            scale = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
            if scale.ndim and scale.ndim < x.ndim:
                scale = scale.view(scale.size(0), *(1,)*(x.ndim - scale.ndim))

        return scale * x

    def preconditioning_loss(self, loss, sigma):
        if self.preconditioning == "song":
            weight = 1. / sigma**2
        if self.preconditioning == "karras":
            weight = (sigma**2 + self.sigma_data**2) / (sigma + self.sigma_data)**2
        if self.preconditioning == "karras_eloi":
            weight = 1.

        return weight * loss


    def sample_time(self, x):
        if self.preconditioning == "song":
            t = torch.rand(x.shape[0], device=x.device) * (self.sde.T - self.t_eps) + self.t_eps
        if self.preconditioning == "karras":
            log_sigma = self.p_mean + self.p_std * torch.randn(x.shape[0], device=x.device)
            sigma = self.t_eps + torch.exp(log_sigma)
            t = self.sde._inverse_std(sigma) #identity in EDM SDE
        if self.preconditioning == "karras_eloi":
            a = torch.rand(x.shape[0], device=x.device)
            sigma = (self.sde.sigma_max**(1/self.sde.rho) + a*(self.sde.sigma_min**(1/self.sde.rho) - self.sde.sigma_max**(1/self.sde.rho)))**self.sde.rho
            t = self.sde._inverse_std(sigma) #identity in EDM SDE

        return t

    def forward(self, x, t, score_conditioning, **kwargs):
        dnn_input = torch.cat([x] + score_conditioning, dim=1) #b,n_input*d,f,t
        dnn_input = self.preconditioning_input(dnn_input, t)
        noise_input = self.preconditioning_noise(t)
        dnn_output = self.dnn(dnn_input, noise_input)
        output = self.preconditioning_output(dnn_output, t)
        skip = self.preconditioning_skip(x, t)
        tweedie_denoiser = skip + output
        
        return tweedie_denoiser

    def _step(self, batch, batch_idx):
        if len(batch) == 1: #In case we use a dataset with only clean speech
            x, y = batch, None
        elif len(batch) == 2:
            x, y = batch

        t = self.sample_time(x)
        mean, std = self.sde.marginal_prob(x, t, y)
        z = torch.randn_like(x)
        if std.ndim < x.ndim:
            std = std.view(*std.size(), *((1,)*(x.ndim - std.ndim)))
        sigma = std
        perturbed_data = mean + sigma * z

        score_conditioning = []
        tweedie_denoiser = self(perturbed_data, t, score_conditioning=score_conditioning, sde_input=y)

        err = tweedie_denoiser - x
        loss = self._loss(err, sigma)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True, batch_size=self.data_module.batch_size)
        if (batch_idx + 1) % 100 == 0:
            self.log_loss_sigma()
             
        return loss

    def log_loss_sigma(self):
        log_loss = self.loss_logger_diff.log_t_bins()
        figure = plot_loss_by_sigma(log_loss, log_x=True, freescale=False)
        self.logger.experiment.add_figure(f"Epoch={self.current_epoch}/LossSigma", [figure])
        self.loss_logger_diff.reset_loss()

    def validation_step(self, batch, batch_idx, discriminative=False, sr=16000):
        loss = self._step(batch, batch_idx)
        self.log('valid_loss', loss, on_step=False, on_epoch=True, batch_size=self.data_module.batch_size)

        if batch_idx == 0 and self.num_eval_files != 0:
            # Inverse problem evaluation
            self.run_evaluation(sr=sr, discriminative=discriminative)
            # Unconditional sampling
            self.run_unconditional_sampling(batch[0]*0, num_samples_unconditional=self.num_samples_unconditional, sampler_type="karras",predictor="euler", corrector="none", scheduler="karras", noise_std=1, smin=self.sde.sigma_min, smax=self.sde.sigma_max, churn=0)

        return loss
    
    def run_evaluation(self, sr: int = 16000, discriminative: bool = False):
        # Evaluate speech enhancement performance
        pesq_est, si_sdr_est, estoi_est, spec, audio = evaluate_model(self, self.num_eval_files, spec=not self.current_epoch%VIS_EPOCHS, audio=not self.current_epoch%VIS_EPOCHS, discriminative=discriminative)
        print(f"PESQ at epoch {self.current_epoch} : {pesq_est:.2f}")
        print(f"SISDR at epoch {self.current_epoch} : {si_sdr_est:.1f}")
        print(f"ESTOI at epoch {self.current_epoch} : {estoi_est:.2f}")
        print('__________________________________________________________________')
        
        self.log('ValidationPESQ', pesq_est, on_step=False, on_epoch=True)
        self.log('ValidationSISDR', si_sdr_est, on_step=False, on_epoch=True)
        self.log('ValidationESTOI', estoi_est, on_step=False, on_epoch=True)

        if audio is not None and self.logger is not None:
            y_list, x_hat_list, x_list = audio
            for idx, (y, x_hat, x) in enumerate(zip(y_list, x_hat_list, x_list)):
                if self.current_epoch == 0:
                    self.logger.experiment.add_audio(f"Epoch={self.current_epoch} Mix/{idx}", (y / torch.max(torch.abs(y))).unsqueeze(-1), sample_rate=sr, global_step=self.current_epoch)
                    self.logger.experiment.add_audio(f"Epoch={self.current_epoch} Clean/{idx}", (x / torch.max(x)).unsqueeze(-1), sample_rate=sr, global_step=self.current_epoch)
                self.logger.experiment.add_audio(f"Epoch={self.current_epoch} Estimate/{idx}", (x_hat / torch.max(torch.abs(x_hat))).unsqueeze(-1), sample_rate=sr, global_step=self.current_epoch)

        if spec is not None and self.logger is not None:
            figures = []
            y_stft_list, x_hat_stft_list, x_stft_list = spec
            for idx, (y_stft, x_hat_stft, x_stft) in enumerate(zip(y_stft_list, x_hat_stft_list, x_stft_list)):
                figures.append(
                    visualize_example(
                    torch.abs(y_stft), 
                    torch.abs(x_hat_stft), 
                    torch.abs(x_stft), return_fig=True))
            self.logger.experiment.add_figure(f"Epoch={self.current_epoch}/Spec", figures)

    def run_unconditional_sampling(self, *args, sr: int = 16000, **kwargs):
        
        figures = []
        samples = self.unconditional_sampling(*args, **kwargs)

        for idx in len(samples):
            self.logger.experiment.add_audio(f"Epoch={self.current_epoch} Unconditional/{idx}", (samples[idx] / torch.max(torch.abs(samples[idx]))).unsqueeze(-1), sample_rate=sr, global_step=self.current_epoch)
            figures.append(
                visualize_one(
                torch.abs(samples[idx]), return_fig=True))
            self.logger.experiment.add_figure(f"Epoch={self.current_epoch}/UnconditionalSpec", figures)
            
    def to(self, *args, **kwargs):
        self.ema.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def get_song_sampler(self, 
        probability_flow,
        predictor_name, scheduler_name, sde_input, N, 
        conditioning, 
        posterior_name, operator, measurement, A, zeta, zeta_schedule,
        corrector_name, r, corrector_steps,
        **kwargs):

        sde = self.sde.copy()
        sde.N = N
        if self.data_module.return_time:
            linearization = lambda x: x
        else:
            linearization = lambda x: self._istft(self._backward_transform(x))
        score_fn = lambda x, t, score_conditioning: self.sde.score_from_tweedie(self(x, t, score_conditioning), x, t, sde_input)
        return sampling.get_song_sampler(
            predictor_name, scheduler_name, sde=sde, score_fn=score_fn, sde_input=sde_input, 
            eps=self.t_eps, probability_flow=probability_flow, conditioning=conditioning, 
            posterior_name=posterior_name, operator=operator, measurement=measurement, A=A, zeta=zeta, zeta_schedule=zeta_schedule, linearization=linearization, 
            corrector_name=corrector_name, r=r, corrector_steps=corrector_steps,
            **kwargs)

    def get_karras_sampler(self, 
        probability_flow,
        predictor_name, scheduler_name, sde_input, N, 
        conditioning, 
        posterior_name, operator, measurement, A, zeta, zeta_schedule,
        noise_std, smin, smax, churn,
        **kwargs):

        sde = self.sde.copy()
        sde.N = N
        if self.data_module.return_time:
            linearization = lambda x: x
        else:
            linearization = lambda x: self._istft(self._backward_transform(x))
        score_fn = lambda x, t, score_conditioning: self.sde.score_from_tweedie(self(x, t, score_conditioning), x, t, sde_input)
        return sampling.get_karras_sampler(
            predictor_name, scheduler_name, sde=sde, score_fn=score_fn, sde_input=sde_input, 
            eps=self.t_eps, probability_flow=probability_flow, conditioning=conditioning, 
            posterior_name=posterior_name, operator=operator, measurement=measurement, A=A, zeta=zeta, zeta_schedule=zeta_schedule, linearization=linearization, 
            noise_std=noise_std, smin=smin, smax=smax, churn=churn,
            **kwargs)

    def train_dataloader(self):
        return self.data_module.train_dataloader()

    def val_dataloader(self):
        return self.data_module.val_dataloader()

    def test_dataloader(self):
        return self.data_module.test_dataloader()

    def setup(self, stage=None):
        return self.data_module.setup(stage=stage)

    def to_audio(self, spec, length=None):
        return self._istft(self._backward_transform(spec), length)

    def _forward_transform(self, spec):
        return self.data_module.spec_fwd(spec)

    def _backward_transform(self, spec):
        return self.data_module.spec_back(spec)

    def _stft(self, sig):
        return self.data_module.stft(sig)

    def _istft(self, spec, length=None):
        return self.data_module.istft(spec, length)

    def enhance(self, y, 
        sampler_type="song", probability_flow=True, N=50, scheduler="linear",
        predictor="euler-maruyama",
        posterior="none", operator="reverberation", A=None, zeta=50., zeta_schedule="lin-increase",
        corrector="ald", r=0.4, corrector_steps=1, 
        noise_std=1.007, smin=0.05, smax=.8, churn=.1,
        **kwargs
    ):
        """
        One-call speech enhancement of noisy speech `y`, for convenience.
        """
        T_orig = y.size(1)

        norm_factor = y.abs().max()
        y = y / norm_factor
        if self.data_module.return_time:
            Y = torch.unsqueeze(y.cuda(), 0)
            Y = pad_time(Y)
        else:
            Y = torch.unsqueeze(self._forward_transform(self._stft(y.cuda())), 0)
            Y = pad_spec(Y)
        if A is not None:
            A = A.cuda()

        if self.condition == "none":
            score_conditioning = []
        elif self.condition == "noisy":
            score_conditioning = [Y]

        if sampler_type == "song":
            sampler = self.get_song_sampler(
                probability_flow=probability_flow,
                predictor_name=predictor, scheduler_name=scheduler, sde_input=Y, N=N,
                conditioning=score_conditioning, 
                posterior_name=posterior, operator=operator, measurement=Y, A=A, zeta=zeta, zeta_schedule=zeta_schedule,
                corrector_name=corrector, r=r, corrector_steps=corrector_steps,
                **kwargs)
        elif sampler_type == "karras":
            sampler = self.get_karras_sampler(
                probability_flow=probability_flow,
                predictor_name=predictor, scheduler_name=scheduler, sde_input=Y, N=N,
                conditioning=score_conditioning, 
                posterior_name=posterior, operator=operator, measurement=Y, A=A, zeta=zeta, zeta_schedule=zeta_schedule,
                noise_std=noise_std, smin=smin, smax=smax, churn=churn,
                **kwargs)
        else:
            print("{} is not a valid sampler type!".format(sampler_type))
        sample = sampler()[0]

        # if kwargs.get("path", None) is not None:
        #     visualize_one(sample.squeeze(), spec_path=kwargs['path'], name="_in_domain")

        if self.data_module.return_time:
            x_hat = sample.squeeze()[..., : T_orig]
        else:
            x_hat = self.to_audio(sample.squeeze(), T_orig)
        x_hat = x_hat * norm_factor
        x_hat = x_hat.squeeze().cpu()
        return x_hat

    def unconditional_sampling(self, reference_tensor, num_samples_unconditional=4,
        sampler_type="song", probability_flow=True, N=50, scheduler="linear",
        predictor="euler-maruyama",
        posterior="none", operator="none", A=None, zeta=50., zeta_schedule="lin-increase",
        corrector="ald", r=0.4, corrector_steps=1, 
        noise_std=1.007, smin=0.05, smax=.8, churn=.1,
        **kwargs
    ):
        """
        One-call unconditional sampling, for convenience.
        """
        A = None
        score_conditioning = []
        operator = "none"
        posterior = "none"
        Y = reference_tensor[: num_samples_unconditional]
        zeta=0

        if sampler_type == "song":
            sampler = self.get_song_sampler(
                probability_flow=probability_flow,
                predictor_name=predictor, scheduler_name=scheduler, sde_input=Y, N=N,
                conditioning=score_conditioning, 
                posterior_name=posterior, operator=operator, measurement=Y, A=A, zeta=zeta, zeta_schedule=zeta_schedule,
                corrector_name=corrector, r=r, corrector_steps=corrector_steps,
                **kwargs)
        elif sampler_type == "karras":
            sampler = self.get_karras_sampler(
                probability_flow=probability_flow,
                predictor_name=predictor, scheduler_name=scheduler, sde_input=Y, N=N,
                conditioning=score_conditioning, 
                posterior_name=posterior, operator=operator, measurement=Y, A=A, zeta=zeta, zeta_schedule=zeta_schedule,
                noise_std=noise_std, smin=smin, smax=smax, churn=churn,
                **kwargs)
        else:
            print("{} is not a valid sampler type!".format(sampler_type))
        sample = sampler()[0]

        if self.data_module.return_time:
            x_hat = sample.squeeze()
        else:
            x_hat = self.to_audio(sample.squeeze() )
        x_hat = x_hat.squeeze().cpu()
        return x_hat