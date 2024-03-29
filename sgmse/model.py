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
from pesq import pesq
from pystoi import stoi
from glob import glob
import shutil

from sgmse import sampling
from sgmse.sdes import SDERegistry, VESDE, EDM
from sgmse.backbones import BackboneRegistry
from sgmse.util.graphics import visualize_example, visualize_one, plot_loss_by_sigma
from sgmse.util.other import pad_spec, pad_time, si_sdr
from sgmse.util.train_utils import SigmaLossLogger
from sgmse.util.fad import FAD
from sgmse.sampling.schedulers import VESongScheduler, EDMScheduler
from sgmse.sampling.operators import ReverberationOperator

VIS_EPOCHS = 5

torch.autograd.set_detect_anomaly(True)

class ScoreModel(pl.LightningModule):
    def __init__(self,
        backbone: str = "ncsnpp", sde: str = "vesde", preconditioning = "song",
        lr: float = 1e-4, ema_decay: float = 0.999,
        t_eps: float = 0., transform: str = 'none', nolog: bool = False,
        num_eval_files: int = 10, loss_type: str = 'mse', data_module_cls = None, 
        condition: str = "none",
        num_unconditional_files: int = 5,
        testset_dir: str = None,
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
        self.num_unconditional_files = num_unconditional_files
        self.sigma_data = kwargs.get("sigma_data", None)

        self.save_hyperparameters(ignore=['nolog'])
        self.data_module = data_module_cls(**kwargs)
        self.condition = condition
        self.nolog = nolog
        
        # Just for logging loss versus sigma
        bins = np.linspace(0, self.sde.T, 20)
        sigma_bins = self.sde._std(torch.from_numpy(self.sde.scheduler.continuous_step(bins))).numpy()
        self.loss_logger = SigmaLossLogger(sigma_bins)
        self.testset_dir = testset_dir or None

    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--lr", type=float, default=1e-4, help="The learning rate")
        parser.add_argument("--ema_decay", type=float, default=0.999, help="The parameter EMA decay constant (0.999 by default)")
        parser.add_argument("--t_eps", type=float, default=0., help="The minimum time (3e-2 by default in Song, but we found it should be 0 in fact)")
        parser.add_argument("--num_eval_files", type=int, default=10, help="Number of files for speech enhancement performance evaluation during training.")
        parser.add_argument("--loss_type", type=str, default="mse", choices=("mse", "mae", "gaussian_entropy", "kristina", "sisdr", "time_mse"), help="The type of loss function to use.")
        parser.add_argument("--condition", default="noisy", choices=["noisy", "none"])
        parser.add_argument("--preconditioning", default="song", choices=["song", "karras"])

        parser.add_argument("--sigma_data", type=float, default=1.7)
        parser.add_argument("--p_mean", type=float, default=-1.2)
        parser.add_argument("--p_std", type=float, default=1.2)
        parser.add_argument("--num_unconditional_files", type=int, default=4, help="Number of generated unconditional samples during evaluation.")
        parser.add_argument("--testset_dir", type=str, default=None, help="Get the test dir for evaluation during the validation step.")

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

    def preconditioning_input(self, t, shape_tensor):
        if self.preconditioning == "song":
            c_in = 1.
        if self.preconditioning == "karras":
            sigma = self.sde._std(t).squeeze()
            c_in = 1/torch.sqrt( self.sigma_data**2 + sigma**2)
            if c_in.ndim and c_in.ndim < shape_tensor.ndim:
                c_in = c_in.view(c_in.size(0), *(1,)*(shape_tensor.ndim - c_in.ndim))

        return c_in

    def preconditioning_noise(self, t):
        if self.preconditioning == "song":
            if not t.ndim:
                t = t.unsqueeze(0)
            c_noise = t
        if self.preconditioning == "song_sigma":
            sigma = self.sde._std(t).squeeze()
            c_noise = sigma.unsqueeze(0)
        if self.preconditioning == "karras":
            sigma = self.sde._std(t).squeeze()
            if not sigma.ndim:
                sigma = sigma.unsqueeze(0)
            c_noise = sigma **.25

        return c_noise

    def preconditioning_output(self, t, shape_tensor):
        if self.preconditioning == "song":
            sigma = self.sde._std(t).squeeze()
            c_out = sigma
        if self.preconditioning == "karras":
            sigma = self.sde._std(t).squeeze()
            c_out = sigma * self.sigma_data / torch.sqrt(self.sigma_data**2 + sigma**2)

        if c_out.ndim and c_out.ndim < shape_tensor.ndim:
            c_out = c_out.view(c_out.size(0), *(1,)*(shape_tensor.ndim - c_out.ndim))

        return c_out

    def preconditioning_skip(self, t, shape_tensor):
        if self.preconditioning == "song":
            c_skip = 1.
        if self.preconditioning == "karras":
            sigma = self.sde._std(t).squeeze()
            c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
            if c_skip.ndim and c_skip.ndim < shape_tensor.ndim:
                c_skip = c_skip.view(c_skip.size(0), *(1,)*(shape_tensor.ndim - c_skip.ndim))

        return c_skip
    
    def preconditioning_target(self, x, n, t):
        ''' As proposed in Karras et al. in the derviations for the loss weighting. Using this formulation assures that the loss is somehow normalized.'''
        c_out = self.preconditioning_output(t, shape_tensor=x)
        c_skip = self.preconditioning_skip(t, shape_tensor=x)

        target= 1/c_out * (x - c_skip * (x+n))
        return target

    def sample_time(self, x):
        a = torch.rand(x.shape[0], device=x.device)
        t = self.sde.scheduler.continuous_step(a)
        return t

    def forward(self, x, t, score_conditioning, **kwargs):
        '''For inference. Return directly tweedie denoiser'''
        input = torch.cat([x] + score_conditioning, dim=1) #b,n_input*d,f,t

        c_in = self.preconditioning_input(t, shape_tensor=x)
        c_out = self.preconditioning_output(t, shape_tensor=x)
        c_skip = self.preconditioning_skip(t, shape_tensor=x)
        noise_input = self.preconditioning_noise(t)

        dnn_input = c_in * input
        dnn_output = self.dnn(dnn_input, noise_input)
        output =  c_out * dnn_output
        skip = c_skip * x

        tweedie_denoiser = skip + output
        
        return tweedie_denoiser

    def forward_train(self, x, t, score_conditioning, **kwargs):
        '''For training only, raw dnn_output'''
        input = torch.cat([x] + score_conditioning, dim=1) #b,n_input*d,f,t

        c_in = self.preconditioning_input(t, shape_tensor=x)
        noise_input = self.preconditioning_noise(t)

        dnn_input = c_in * input
        dnn_output = self.dnn(dnn_input, noise_input)

        return dnn_output

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
        noise = std * z
        perturbed_data = mean + noise

        score_conditioning = []
        dnn_output = self.forward_train(perturbed_data, t, score_conditioning=score_conditioning, sde_input=y)
        target = self.preconditioning_target(x, noise, t)

        err = dnn_output - target
        losses = torch.square(err.abs()).reshape(err.shape[0], -1)
        loss = torch.mean(losses)

        # Log loss vs sigma
        for i in range(len(err)):
            self.loss_logger.write_loss(std[i].item(), losses[i].mean().item(), losses[i].var().item())

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True, batch_size=self.data_module.batch_size, sync_dist=True)
        if (batch_idx + 1) % 100 == 0:
            self.log_loss_sigma()
             
        return loss

    def log_loss_sigma(self):
        log_loss = self.loss_logger.log_t_bins()
        figure = plot_loss_by_sigma(log_loss, log_x=True, freescale=False)
        self.logger.experiment.add_figure(f"Epoch={self.current_epoch}/LossSigma", [figure])
        self.loss_logger.reset_logger()

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self.log('valid_loss', loss, on_step=False, on_epoch=True, batch_size=self.data_module.batch_size, sync_dist=True)

        if isinstance(self.sde, VESDE):
            kwargs_posterior = dict(sampler_type="song", predictor="euler-maruyama", scheduler="ve", probability_flow=True, zeta=7., zeta_schedule="div-sig")
            kwargs_unconditional = kwargs_posterior
        if isinstance(self.sde, EDM):
            kwargs_posterior = dict(sampler_type="karras", predictor="euler-heun-dps", corrector="none", scheduler="edm", noise_std=1, smin=0., smax=0., churn=0., probability_flow=True, zeta=7., zeta_schedule="div-sig")
            kwargs_unconditional = dict(sampler_type="karras", predictor="euler-heun", corrector="none", scheduler="edm", noise_std=1, smin=0., smax=0., churn=0., probability_flow=True)

        if batch_idx == 0:
            if hasattr(self, "testset_dir") and self.testset_dir is not None:
                # Inverse problem evaluation
                self.run_posterior_sampling(**kwargs_posterior)
            # Unconditional sampling
            self.run_unconditional_sampling(**kwargs_unconditional)

        return loss
    
    def run_posterior_sampling(self, _max_vis_samples=10, _vis_epochs=10, **kwargs):
        # Evaluate posterior sampling performance
        x_files = sorted(glob(os.path.join(self.testset_dir, "audio/tt/clean", "*.wav")))
        y_files = sorted(glob(os.path.join(self.testset_dir, "audio/tt/noisy", "*.wav")))
        rir_files = sorted(glob(os.path.join(self.testset_dir, "rir/tt", "*.wav")))
        x_list, y_list, x_hat_list = [], [], []

        for i in range(self.num_eval_files):

            operator = ReverberationOperator(kernel_size=int(1.*self.data_module.sample_rate), stft=False, **self.data_module.stft_kwargs)
            A, rir_sr = torchaudio.load(rir_files[i])
            if rir_sr != self.data_module.sample_rate:
                A = torchaudio.utils.Resample(orig_freq=rir_sr, new_freq=self.data_module.sample_rate)(A)
            operator.load_weights(A.squeeze(0))

            y, sr = torchaudio.load(y_files[i])
            x, sr = torchaudio.load(x_files[i])
            y = y[..., : int(4. * sr)] #Avoid GPU memory overload because of DPS gradients
            x = x[..., : int(4. * sr)] #Avoid GPU memory overload because of DPS gradients
            if sr != self.data_module.sample_rate:
                y = torchaudio.utils.Resample(orig_freq=sr, new_freq=self.data_module.sample_rate)(y)
                x = torchaudio.utils.Resample(orig_freq=sr, new_freq=self.data_module.sample_rate)(x)

            x_hat = self.enhance(y, operator=operator, A=A, **kwargs)

            x_list.append(x.squeeze())
            y_list.append(y.squeeze())
            x_hat_list.append(x_hat.squeeze())
        
        self.log_metrics(x_list, y_list, x_hat_list, _max_vis_samples=_max_vis_samples)
        self.log_audio(x_list, y_list, x_hat_list, _max_vis_samples=_max_vis_samples, _vis_epochs=_vis_epochs)
        self.log_spec(x_list, y_list, x_hat_list, _max_vis_samples=_max_vis_samples, _vis_epochs=_vis_epochs)

    def run_unconditional_sampling(self, _len_generation=5., _max_vis_samples=10, _vis_epochs=10, **kwargs):
        figures = []
        if self.current_epoch%_vis_epochs==0 and _max_vis_samples and self.logger is not None:

            gt_dir = os.path.join(self.logger.log_dir, ".fad_cache/gt")
            generated_dir = os.path.join(self.logger.log_dir, ".fad_cache/generated")
            os.makedirs(gt_dir, exist_ok=True)
            os.makedirs(generated_dir, exist_ok=True)
            gt_files = sorted(glob(os.path.join(self.testset_dir, "audio", "tt", "clean", "*.wav")))

            for idx in range(self.num_unconditional_files):
                reference_tensor = torch.zeros(1, int(_len_generation*self.data_module.sample_rate))
                x_hat = self.unconditional_sampling(reference_tensor, **kwargs).squeeze()
                x_hat_stft = self._stft(x_hat)

                self.logger.experiment.add_audio(f"Epoch={self.current_epoch} Unconditional/{idx}", (x_hat / torch.max(torch.abs(x_hat))), sample_rate=self.data_module.sample_rate, global_step=self.current_epoch)
                figures.append(
                    visualize_one(
                    torch.abs(x_hat_stft), return_fig=True))
                self.logger.experiment.add_figure(f"Epoch={self.current_epoch}/UnconditionalSpec", figures)
            
                torchaudio.save(os.path.join(self.logger.log_dir, ".fad_cache", "generated", f"{idx}.wav"), (x_hat / torch.max(torch.abs(x_hat))).type(torch.float32).cpu().squeeze().unsqueeze(0), self.data_module.sample_rate)
                x, sr = torchaudio.load(gt_files[idx])
                if sr != self.data_module.sample_rate:
                    x = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.data_module.sample_rate)(x)
                torchaudio.save(os.path.join(self.logger.log_dir, ".fad_cache", "gt", f"{idx}.wav"), (x / torch.max(torch.abs(x))), self.data_module.sample_rate)

            self.log_fad(gt_dir, generated_dir)

    def run_supervised_enhancement(self, batch, _max_vis_samples=10, _vis_epochs=10, **kwargs):
        # Evaluate speech enhancement performance, for conditional models such as SGMSE+ and StoRM
        x, y = batch
        
        x_hat = self.enhance(y, **kwargs)
        self.log_metrics(x, y, x_hat, _max_vis_samples=_max_vis_samples)
        self.log_audio(x, y, x_hat, _max_vis_samples=_max_vis_samples, _vis_epochs=_vis_epochs)
        self.log_spec(x, y, x_hat, _max_vis_samples=_max_vis_samples, _vis_epochs=_vis_epochs)

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

        sde = self.sde
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

        sde = self.sde
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

    def get_reddiff_sampler(self,
        scheduler_name, sde_input, N, 
        conditioning, 
        operator, measurement, A, zeta, zeta_schedule,
        optimizer_name, lr, stochastic_std,
        **kwargs):

        sde = self.sde
        sde.N = N
        if self.data_module.return_time:
            linearization = lambda x: x
        else:
            linearization = lambda x: self._istft(self._backward_transform(x))

        tweedie_fn = self.forward
        score_fn = lambda x, t, score_conditioning: self.sde.score_from_tweedie(self(x, t, score_conditioning), x, t, sde_input)

        return sampling.get_reddiff_sampler(
            scheduler_name=scheduler_name, sde=sde, 
            tweedie_fn=tweedie_fn,
            # score_fn=score_fn,
            sde_input=sde_input, 
            conditioning=conditioning, 
            operator=operator, measurement=measurement, A=A, zeta=zeta, zeta_schedule=zeta_schedule, linearization=linearization,
            optimizer_name=optimizer_name, lr=lr, stochastic_std=stochastic_std,
            **kwargs
        )

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
        sampler_type="song", probability_flow=True, N=50, scheduler="ve",
        predictor="euler-maruyama",
        posterior="dps", operator=None, A=None, zeta=60., zeta_schedule="saw-tooth-increase",
        corrector="ald", r=0.4, corrector_steps=1, #For Song
        noise_std=1.007, smin=0.05, smax=.8, churn=.1, #For Karras
        optimizer="adam", lr=1e-1, stochastic_std=1e-4, #For RED-Diff
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
        elif sampler_type == "red-diff":
            sampler = self.get_reddiff_sampler(
                scheduler_name=scheduler, sde_input=Y, N=N, 
                conditioning=score_conditioning, 
                operator=operator, measurement=Y, A=A, zeta=zeta, zeta_schedule=zeta_schedule,
                optimizer_name=optimizer, lr=lr, stochastic_std=stochastic_std,
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

    def unconditional_sampling(self, reference_tensor, 
        sampler_type="song", probability_flow=True, N=50, scheduler="ve",
        predictor="euler-maruyama",
        corrector="ald", r=0.4, corrector_steps=1, 
        noise_std=1.007, smin=0.05, smax=.8, churn=.1,
        **kwargs
    ):
        """
        One-call unconditional sampling, for convenience.
        """
        score_conditioning = []
        Y = torch.unsqueeze(self._stft(reference_tensor.cuda()), 0)
        Y = pad_spec(Y)
        
        if sampler_type == "song":
            sampler = self.get_song_sampler(
                probability_flow=probability_flow,
                predictor_name=predictor, scheduler_name=scheduler, sde_input=Y, N=N,
                conditioning=score_conditioning, 
                posterior_name="none", operator=None, measurement=Y, A=None, zeta=0., zeta_schedule="none",
                corrector_name=corrector, r=r, corrector_steps=corrector_steps,
                **kwargs)
        elif sampler_type == "karras":
            sampler = self.get_karras_sampler(
                probability_flow=probability_flow,
                predictor_name=predictor, scheduler_name=scheduler, sde_input=Y, N=N,
                conditioning=score_conditioning, 
                posterior_name="none", operator=None, measurement=Y, A=None, zeta=0., zeta_schedule="none",
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
    
    def log_metrics(self, x, y, x_hat, _max_vis_samples=10):

        _si_sdr, _pesq, _estoi = np.zeros((min(len(x), _max_vis_samples),)), np.zeros((min(len(x), _max_vis_samples),)), np.zeros((min(len(x), _max_vis_samples),))

        for i in range(min(len(x), _max_vis_samples)):
            _si_sdr[i] = si_sdr(x[i], x_hat[i])
            if self.data_module.sample_rate == 16000:
                x_resampled = x[i].cpu().numpy()
                x_hat_resampled = x_hat[i].cpu().numpy()
            else:
                print("Not 16kHz. Resampling to 16kHz for PESQ and (E)STOI")
                x_resampled = torchaudio.transforms.Resample(orig_freq=self.data_module.sample_rate, new_freq=16000)(x[i]).cpu().numpy()
                x_hat_resampled = torchaudio.transforms.Resample(orig_freq=self.data_module.sample_rate, new_freq=16000)(x_hat[i]).cpu().numpy()
            try:
                _pesq[i] = pesq(16000, x_resampled, x_hat_resampled, 'wb') 
            except ValueError:
                _pesq[i] = 0.
            _estoi[i] = stoi(x_resampled, x_hat_resampled, 16000, extended=True)

        print(f"PESQ at epoch {self.current_epoch} : {_pesq.mean():.2f}")
        print(f"SISDR at epoch {self.current_epoch} : {_si_sdr.mean():.1f}")
        print(f"ESTOI at epoch {self.current_epoch} : {_estoi.mean():.2f}")
        print('__________________________________________________________________')
        
        self.log('ValidationPESQ', _pesq.mean(), on_step=False, on_epoch=True, sync_dist=True)
        self.log('ValidationSISDR', _si_sdr.mean(), on_step=False, on_epoch=True, sync_dist=True)
        self.log('ValidationESTOI', _estoi.mean(), on_step=False, on_epoch=True, sync_dist=True)

    def log_fad(self, gt_dir, generated_dir):
        
        _fad = FAD(gt_dir, generated_dir)
        print(f"FAD at epoch {self.current_epoch} : {_fad:.2f}")
        self.log('ValidationFAD', _fad, on_step=False, on_epoch=True, sync_dist=True)

    def log_audio(self, x, y, x_hat, _max_vis_samples, _vis_epochs):

        if self.current_epoch%_vis_epochs==0 and _max_vis_samples and self.logger is not None:
            for i in range(min(len(x), _max_vis_samples)):
                if self.current_epoch == 0:
                    self.logger.experiment.add_audio(f"Epoch={self.current_epoch} Clean/{i}", (x[i] / torch.max(x[i])).unsqueeze(-1), sample_rate=self.data_module.sample_rate, global_step=self.current_epoch)
                    self.logger.experiment.add_audio(f"Epoch={self.current_epoch} Mix/{i}", (y[i] / torch.max(torch.abs(y[i]))).unsqueeze(-1), sample_rate=self.data_module.sample_rate, global_step=self.current_epoch)
                self.logger.experiment.add_audio(f"Epoch={self.current_epoch} Estimate/{i}", (x_hat[i] / torch.max(torch.abs(x_hat[i]))).unsqueeze(-1), sample_rate=self.data_module.sample_rate, global_step=self.current_epoch)

    def log_spec(self, x, y, x_hat, _max_vis_samples=10, _vis_epochs=10):

        if self.current_epoch%_vis_epochs==0 and _max_vis_samples and self.logger is not None:
            figures = []
            for i in range(min(len(x), _max_vis_samples)):
                figures.append(
                    visualize_example(
                    torch.abs(self._stft(y[i])), 
                    torch.abs(self._stft(x_hat[i])), 
                    torch.abs(self._stft(x[i])), return_fig=True))
            self.logger.experiment.add_figure(f"Epoch={self.current_epoch}/Spec", figures)
