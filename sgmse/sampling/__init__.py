# Adapted from https://github.com/yang-song/score_sde_pytorch/blob/1618ddea340f3e4a2ed7852a0694a809775cf8d0/sampling.py
"""Various sampling methods."""
from scipy import integrate
import torch
import numpy as np
import tqdm
import torchaudio
import matplotlib.pyplot as plt
from .predictors import Predictor, PredictorRegistry
from .correctors import Corrector, CorrectorRegistry
from .posteriors import Posterior, PosteriorRegistry
from .operators import LinearOperator, OperatorRegistry
from .schedulers import Scheduler, SchedulerRegistry, LinearScheduler, KarrasScheduler

__all__ = [
    'PredictorRegistry', 'CorrectorRegistry', 'PosteriorRegistry', 'OperatorRegistry', 'SchedulerRegistry'
    'Predictor', 'Corrector', 'Posterior', 'Operator', 'Scheduler',
    'get_sampler'
]


def to_flattened_numpy(x):
    """Flatten a torch tensor `x` and convert it to numpy."""
    return x.detach().cpu().numpy().reshape((-1,))


def from_flattened_numpy(x, shape):
    """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
    return torch.from_numpy(x.reshape(shape))

def pick_zeta_schedule(schedule, t, sigma_t, zeta, clip=2500):
    if schedule == "none":
        return None
    if schedule == "const":
        zeta_t = zeta
    if schedule == "lin-increase":
        zeta_t = zeta * t
    if schedule == "lin-decrease":
        zeta_t = zeta * (1-t)
    if schedule == "half-cycle":
        zeta_t = zeta * np.sin(np.pi * t)
    if schedule == "sqrt-increase":
        zeta_t = zeta * np.sqrt(1e-10 + t)
    if schedule == "exp-increase":
        zeta_t = zeta * np.exp(t)
    if schedule == "log-increase":
        zeta_t = zeta * np.log(1+1e10+t)
    if schedule == "div-sig":
        zeta_t = zeta / sigma_t
    if schedule == "div-sig-square":
        zeta_t = zeta / sigma_t**2
    if schedule == "saw-tooth-increase":
        max_step = .9
        if t < max_step: #ramp from 0 to zeta0 in rho_max
            zeta_t = zeta/max_step * t
        else:
            zeta_t = zeta + zeta * (max_step - t)/(1-max_step)

    return min(zeta_t, clip)


def get_song_sampler(
    predictor_name = "euler-maruyama", scheduler_name = "ve-song", sde = "ve", score_fn = None, sde_input = None, 
    eps = 3e-2, probability_flow = True,  conditioning = "none",
    posterior_name = "none", operator = "none", measurement = None, A = None, zeta = 50, zeta_schedule = "lin-increase", linearization = None,
    corrector_name = "ald", r = .5, corrector_steps = 1, denoise = True,
    **kwargs

):
    """Create a Predictor-Corrector (PC) sampler with Posterior Sampling.

    Args:
        predictor_name: The name of a registered `sampling.Predictor`.
        corrector_name: The name of a registered `sampling.Corrector`.
        sde: An `sdes.SDE` object representing the forward SDE.
        score_fn: A function (typically learned model) that predicts the score.
        y: A `torch.Tensor`, representing the (non-white-)noisy starting point(s) to condition the prior on.
        denoise: If `True`, add one-step denoising to the final samples.
        eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
        r: The r to use for the corrector. 0.1 by default, and ignored for `NoneCorrector`.
        N: The number of reverse sampling steps. If `None`, uses the SDE's `N` property by default.

    Returns:
        A sampling function that returns samples and the number of function evaluations during sampling.
    """
    predictor_cls = PredictorRegistry.get_by_name(predictor_name)
    corrector_cls = CorrectorRegistry.get_by_name(corrector_name)
    scheduler_cls = SchedulerRegistry.get_by_name(scheduler_name)
    posterior_cls = PosteriorRegistry.get_by_name(posterior_name)
    predictor = predictor_cls(sde, score_fn, probability_flow=probability_flow)
    corrector = corrector_cls(sde, score_fn, r=r, n_steps=corrector_steps)
    posterior = posterior_cls(sde, operator, linearization, zeta=zeta)
    scheduler = scheduler_cls(eps=eps, **sde.__dict__, **kwargs)
    
    def song_sampler():
        """The Posterior sampler function."""
        zeta0 = posterior.zeta
        xt = sde.prior_sampling(sde_input.shape, sde_input, **kwargs).to(sde_input.device)
        At = A
        distance = torch.Tensor([.0])
        timesteps = scheduler.timesteps().to(xt.device)
        pbar = tqdm.tqdm(list(range(sde.N)))

        for i in pbar:
            dt = timesteps[i+1] - timesteps[i] # dt < 0 (time flowing in reverse)
            t = torch.ones(sde_input.shape[0], device=sde_input.device) * timesteps[i]
            if posterior_name != "none":
                posterior.zeta = pick_zeta_schedule(zeta_schedule, t.cpu().item(), sde._std(t).cpu().item(), zeta0)

            # corrector
            with torch.no_grad():
                xt, xt_mean = corrector.update_fn(xt, t, dt, conditioning=conditioning, sde_input=sde_input)

            # predictor
            grad_required = posterior_name == "dps" or (posterior_name == "switching" and timesteps[i].item() > kwargs["sw"]) or (posterior_name == "reverse-switching" and timesteps[i].item() < kwargs["sw"])
            xt = xt.requires_grad_(grad_required)
            with torch.set_grad_enabled(grad_required):
                xt, xt_mean, score = predictor.update_fn(xt, t, dt, conditioning=conditioning, sde_input=sde_input)

            # posterior
            if i < sde.N - 1:
                xt, At, distance, yt, x0t_linear = posterior.update_fn(xt, t, dt, measurement=measurement, sde_input=sde_input, score=score, A=At, **kwargs)
                xt, xt_mean, score = xt.detach(), xt_mean.detach(), score.detach()
                pbar.set_postfix({'distance': distance.item()}, refresh=False)

        x_result = xt_mean if (denoise and sde.N) else xt
        ns = sde.N * (corrector.n_steps + int(grad_required) + 1)
        return x_result, ns, At, distance
    return song_sampler




def get_karras_sampler(
    predictor_name = "euler-heun", scheduler_name = "edm", sde = "edm", score_fn = None, sde_input = None, 
    eps = 0., probability_flow = True,  conditioning = "none", 
    posterior_name = "none", operator = "none", measurement = None, A = None, zeta = 50, zeta_schedule = "lin-increase", linearization = None,
    noise_std = 1.007, smin = 0.05, smax = .8, churn = .1,
    **kwargs
):
    """Create a Predictor-Corrector (PC) sampler with Posterior Sampling.

    Args:
        predictor_name: The name of a registered `sampling.Predictor`.
        corrector_name: The name of a registered `sampling.Corrector`.
        sde: An `sdes.SDE` object representing the forward SDE.
        score_fn: A function (typically learned model) that predicts the score.
        y: A `torch.Tensor`, representing the (non-white-)noisy starting point(s) to condition the prior on.
        denoise: If `True`, add one-step denoising to the final samples.
        eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
        r: The r to use for the corrector. 0.1 by default, and ignored for `NoneCorrector`.
        N: The number of reverse sampling steps. If `None`, uses the SDE's `N` property by default.

    Returns:
        A sampling function that returns samples and the number of function evaluations during sampling.
    """
    predictor_cls = PredictorRegistry.get_by_name(predictor_name)
    scheduler_cls = SchedulerRegistry.get_by_name(scheduler_name)
    posterior_cls = PosteriorRegistry.get_by_name(posterior_name)
    predictor = predictor_cls(sde, score_fn, probability_flow=probability_flow)
    posterior = posterior_cls(sde, operator, linearization, zeta=zeta)
    scheduler = scheduler_cls(eps=eps, **sde.__dict__, **kwargs)
    
    def karras_sampler():
        """The Posterior sampler function."""
        zeta0 = posterior.zeta
        xt = sde.prior_sampling(sde_input.shape, sde_input, **kwargs).to(sde_input.device)
        At = A
        distance = torch.Tensor([.0])
        timesteps = scheduler.timesteps().to(xt.device)
        pbar = tqdm.tqdm(list(range(sde.N)))

        for i in pbar:
            z = noise_std * torch.randn_like(xt)
            gamma = min(churn/sde.N, np.sqrt(2)-1.) if (timesteps[i] > smin and timesteps[i] < smax) else 0.
            t_overnoised = timesteps[i]*(1 + gamma)
            # dt = timesteps[i+1] - timesteps[i] # dt < 0 (time flowing in reverse)
            dt = timesteps[i+1] - t_overnoised # Suggested by Eloi
            if posterior_name != "none":
                posterior.zeta = pick_zeta_schedule(zeta_schedule, t_overnoised.cpu().item(), sde._std(t_overnoised).cpu().item(), zeta0)
            xt = xt + torch.sqrt(t_overnoised**2 - timesteps[i]**2) * torch.ones_like(xt) * z

            # predictor
            grad_required = posterior_name == "dps" or (posterior_name == "switching" and timesteps[i].item() > kwargs["sw"]) or (posterior_name == "reverse-switching" and timesteps[i].item() < kwargs["sw"])
            xt = xt.requires_grad_(grad_required)
            with torch.set_grad_enabled(grad_required):
                xt, xt_mean, score = predictor.update_fn(xt, t_overnoised * torch.ones(sde_input.shape[0], device=sde_input.device), dt, conditioning=conditioning, sde_input=sde_input, **kwargs)
    
            # posterior
            if i < sde.N - 1:
                t = torch.ones(sde_input.shape[0], device=sde_input.device) * timesteps[i]
                xt, At, distance, yt, x0t_linear = posterior.update_fn(xt, t, dt, measurement=measurement, sde_input=sde_input, score=score, A=At, **kwargs)
                xt, xt_mean, score = xt.detach(), xt_mean.detach(), score.detach()
                pbar.set_postfix({'distance': distance.item()}, refresh=False)

        x_result = xt
        ns = sde.N * (int(grad_required) + 1)
        return x_result, ns, At, distance
    return karras_sampler



