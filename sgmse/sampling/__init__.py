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
from .schedulers import Scheduler, SchedulerRegistry
from .optimizers import get_optimizer

from sgmse.util.graphics import visualize_one
from sgmse.util.other import pad_spec

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

def pick_zeta_schedule(schedule, t, zeta, clip=50_000):
    if schedule == "none":
        return None
    if schedule == "constant":
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
        zeta_t = zeta / t
    if schedule == "div-sig-square":
        zeta_t = zeta / t**2
    if schedule == "saw-tooth-increase":
        max_step = .9
        if t < max_step: #ramp from 0 to zeta0 in rho_max
            zeta_t = zeta/max_step * t
        else:
            zeta_t = zeta + zeta * (max_step - t)/(1-max_step)

    return min(zeta_t, clip)


def get_song_sampler(
    predictor_name = "euler-maruyama", scheduler_name = "ve-song", sde = None, score_fn = None, sde_input = None, 
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
    corrector = corrector_cls(sde, score_fn, probability_flow=probability_flow, r=r, n_steps=corrector_steps)
    posterior = posterior_cls(sde, operator, linearization, zeta=zeta)
    scheduler = scheduler_cls(eps=eps, **sde.__dict__, **kwargs)
    
    def song_sampler():
        """The Posterior sampler function."""
        zeta0 = posterior.zeta
        xt = sde.prior_sampling(sde_input.shape, sde_input, **kwargs).to(sde_input.device)
        At = A
        distance = torch.Tensor([.0])
        timesteps = scheduler.reverse_timesteps(sde.T).to(xt.device)
        pbar = tqdm.tqdm(list(range(sde.N)))

        for i in pbar:
            dt = timesteps[i+1] - timesteps[i] # dt < 0 (time flowing in reverse)
            t = torch.ones(sde_input.shape[0], device=sde_input.device) * timesteps[i]
            if posterior_name != "none":
                # posterior.zeta = pick_zeta_schedule(zeta_schedule, t.cpu().item(), sde._std(t).cpu().item(), zeta0)
                posterior.zeta = pick_zeta_schedule(zeta_schedule, min(1., t.cpu().item() / scheduler.continuous_step(sde.T)), zeta0) #weird fix for now

            # corrector
            with torch.no_grad():
                xt, xt_mean = corrector.update_fn(xt, t, dt, conditioning=conditioning, sde_input=sde_input)

            # predictor
            grad_required = posterior.grad_required(t)
            xt = xt.requires_grad_(grad_required)
            with torch.set_grad_enabled(grad_required):
                xt, xt_mean, score = predictor.update_fn(xt, t, dt, conditioning=conditioning, sde_input=sde_input)

            # posterior
            if i < sde.N - 1:
                with torch.set_grad_enabled(grad_required):
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
    if predictor_name=="euler-heun-dps": #Integrate DPS in Predictor, as one DPS step is used per score estimation.
        predictor = predictor_cls(sde, score_fn,  operator, linearization, zeta, probability_flow=False)
        posterior = predictor #For zeta updates
    else:
        predictor = predictor_cls(sde, score_fn,probability_flow=probability_flow)
        posterior = posterior_cls(sde, operator, linearization, zeta=zeta)
    scheduler = scheduler_cls(eps=eps, **sde.__dict__, **kwargs)
    
    def karras_sampler():
        """The Posterior sampler function."""
        zeta0 = posterior.zeta
        xt = sde.prior_sampling(sde_input.shape, sde_input, **kwargs).to(sde_input.device)
        At = A
        distance = torch.Tensor([.0])
        timesteps = scheduler.reverse_timesteps(sde.T).to(xt.device)
        pbar = tqdm.tqdm(list(range(sde.N)))

        for i in pbar:
            z = noise_std * torch.randn_like(xt)
            gamma = min(churn/sde.N, np.sqrt(2)-1.) if (not(probability_flow) and timesteps[i] > smin and timesteps[i] < smax) else 0.
            t_overnoised = timesteps[i]*(1 + gamma) * torch.ones(sde_input.shape[0], device=sde_input.device)
            if posterior_name != "none":
                posterior.zeta = pick_zeta_schedule(zeta_schedule, t_overnoised.cpu().item(), zeta0)
            dt = timesteps[i+1] - timesteps[i]*(1 + gamma) # dt < 0 (time flowing in reverse)
            if gamma > 0:
                xt = xt + timesteps[i] * torch.sqrt((1 + gamma)**2 - 1) * torch.ones_like(xt) * z

            # predictor
            grad_required = posterior.grad_required(timesteps[i].item(), **kwargs)
            xt = xt.requires_grad_(grad_required)
            with torch.set_grad_enabled(grad_required):
                if predictor_name=="euler-heun-dps":
                    xt, xt_mean, score, At, distance = predictor.update_fn(xt, t_overnoised, dt, conditioning=conditioning, sde_input=sde_input,measurement=measurement, A=At, **kwargs)
                else:
                    xt, xt_mean, score = predictor.update_fn(xt, t_overnoised, dt, conditioning=conditioning, sde_input=sde_input, **kwargs)
    
            # posterior
            if i < sde.N - 1 and predictor_name != "euler-heun-dps" and posterior.zeta > 0.:
                with torch.set_grad_enabled(grad_required):
                    xt, At, distance, yt, x0t_linear = posterior.update_fn(xt, t_overnoised, dt, measurement=measurement, sde_input=sde_input, score=score, A=At, **kwargs)
                xt, xt_mean, score = xt.detach(), xt_mean.detach(), score.detach()
            pbar.set_postfix({'distance': distance.item()}, refresh=False)

            if grad_required:
                xt, xt_mean, score = xt.detach(), xt_mean.detach(), score.detach()

        x_result = xt
        ns = sde.N * (int(grad_required) + 1)
        return x_result, ns, At, distance
    return karras_sampler



def get_reddiff_sampler(
    scheduler_name = "edm", sde = "edm", 
    # score_fn = None, 
    tweedie_fn = None, 
    sde_input = None, 
    conditioning = "none", 
    operator = "none", measurement = None, A = None, zeta = 0.25, zeta_schedule = "lin-increase", linearization = None,
    optimizer_name = "adam", lr = 0.1, stochastic_std = 0.,
    **kwargs
):
    """Create a RED-Diff stochastic optimizer, as in:
    
    A VARIATIONAL PERSPECTIVE ON SOLVING INVERSE PROBLEMS WITH DIFFUSION MODELS
    Morteza Mardani, Jiaming Song, Jan Kautz, Arash Vahdat
    2023

    Adapted from https://github.com/NVlabs/RED-diff/blob/master/algos/reddiff.py

    Args:
        sde: An `sdes.SDE` object representing the forward SDE.
        score_fn: A function (typically learned model) that predicts the score.
        y: A `torch.Tensor`, representing the (non-white-)noisy starting point(s) to condition the prior on.

    Returns:
        A sampling function that returns samples and the number of function evaluations during sampling.
    """
    scheduler_cls = SchedulerRegistry.get_by_name(scheduler_name)
    scheduler = scheduler_cls(eps=0., **sde.__dict__, **kwargs)
    
    def REDDiff_optimizer():
        """The stochastic optimization function."""

        # x0 = torch.randn_like(measurement)
        x0 = torch.zeros_like(measurement)

        # x0 = measurement + torch.randn_like(measurement)
        # x0 = measurement
        # x0 = torch.linalg.pinv(A) * measurement + torch.randn_like(measurement)
        # x0 = tweedie_model._stft( operator.pinv(tweedie_model._istft(measurement.squeeze()), A.squeeze())).unsqueeze(0).unsqueeze(0)
        # x0 = pad_spec(x0)
        # x0 = sde.prior_sampling(sde_input.shape, sde_input, **kwargs).to(sde_input.device)

        At = A

        timesteps = scheduler.reverse_timesteps(sde.T).to(x0.device)
        pbar = tqdm.tqdm(list(range(sde.N)))
        mu = torch.autograd.Variable(x0, requires_grad=True)
        optimizer = get_optimizer(optimizer_name, params=[mu], lr=lr)

        for i in pbar:

            # Slightly perturb curent prediction (why? Not in the paper? Check)
            z0 = torch.randn_like(mu)
            mu_perturbed = mu + stochastic_std * z0

            # Score matching term
            t = timesteps[i] * torch.ones(sde_input.shape[0], device=sde_input.device)
            mean, std = sde.marginal_prob(mu_perturbed, t, sde_input)
            z = torch.randn_like(mu_perturbed)
            if std.ndim < mu_perturbed.ndim:
                std = std.view(*std.size(), *((1,)*(mu_perturbed.ndim - std.ndim)))
            xt = mean + std * z

            # Option 1: with Tweedie
            # RED score matching regularization (using the Tweedie-wise eq. 10 rather than the score-wise eq. 9)
            tweedie = tweedie_fn(xt, t, conditioning).detach()
            # loss_score = torch.mul((tweedie - mu_perturbed).detach(), mu_perturbed).mean()
            loss_score = (tweedie - mu_perturbed).abs().square().sum()  # Try same as Eloi
            weight_score = zeta 

            # # Option 2: with score
            # score = score_fn(xt, t, conditioning).detach()
            # loss_score = torch.mul((score * std - z).detach(), mu_perturbed).mean()
            # weight_score = zeta * std

            # Posterior observation term
            measurement_linear = linearization(measurement.squeeze(0)).unsqueeze(0)
            mu_perturbed_linear = linearization(mu_perturbed.squeeze(0)).unsqueeze(0)
            operator.load_weights(At.squeeze(0))
            measurement_estimated_linear = operator.forward(mu_perturbed_linear)
            # loss_obs = (measurement_linear - measurement_estimated_linear).abs().square().mean() / 2
            loss_obs = (measurement_linear - measurement_estimated_linear).abs().square().sum() # Try same as Eloi

            if i % 50 == 0:
                # print(timesteps[i], std.item())
                # visualize_one(tweedie, spec_path=".test", name=f"tweedie_{i}")
                visualize_one(mu, spec_path=".test", name=f"mu_{i}")

            # n_fft = 512
            # hop_length = 128
            # stft_kwargs = {"n_fft": n_fft, "hop_length": hop_length, "window": torch.hann_window(n_fft).cuda(), "center": True, "return_complex": True}
            # measurement_estimated = torch.stft(measurement_estimated_linear.squeeze(), **stft_kwargs)
            # measurement_verif = torch.stft(measurement_linear.squeeze(), **stft_kwargs)
            # visualize_one(measurement_estimated, spec_path=".test", name=f"y_hat_{t.cpu().item()}")
            # visualize_one(measurement_verif, spec_path=".test", name=f"y_{t.cpu().item()}")

            # print(loss_obs, loss_score)

            # Weight losses and optimizer step
            loss = loss_obs + weight_score * loss_score
            loss = (1 - weight_score) * loss_obs + weight_score * loss_score # Same as Eloi




            # Optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Log distance to measurement in the same fashion as other samplers
            pbar.set_postfix({'distance': torch.linalg.norm(measurement_linear - measurement_estimated_linear).item()}, refresh=False)

        ns = sde.N
        return mu, ns, At, loss_obs
    
    return REDDiff_optimizer