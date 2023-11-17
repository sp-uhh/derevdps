"""
Abstract SDE classes, Reverse SDE, and VE/VP SDEs.

Taken and adapted from https://github.com/yang-song/score_sde_pytorch/blob/1618ddea340f3e4a2ed7852a0694a809775cf8d0/sde_lib.py
"""
import abc
from email.policy import default
import warnings

import numpy as np
from sgmse.util.tensors import batch_broadcast
import torch

from sgmse.util.registry import Registry
import os

SDERegistry = Registry("SDE")


class SDE(abc.ABC):
    """SDE abstract class. Functions are designed for a mini-batch of inputs."""

    def __init__(self, N):
        """Construct an SDE.

        Args:
            N: number of discretization time steps.
        """
        super().__init__()
        self.N = N

    @property
    @abc.abstractmethod
    def T(self):
        """End time of the SDE."""
        pass

    @abc.abstractmethod
    def sde(self, x, t, *args, **kwargs):
        pass

    @abc.abstractmethod
    def marginal_prob(self, x, t, *args):
        """Parameters to determine the marginal distribution of the SDE, $p_t(x|args)$."""
        pass

    @abc.abstractmethod
    def prior_sampling(self, shape, *args, **kwargs):
        """Generate one sample from the prior distribution, $p_T(x|args)$ with shape `shape`."""
        pass

    @abc.abstractmethod
    def prior_logp(self, z):
        """Compute log-density of the prior distribution.

        Useful for computing the log-likelihood via probability flow ODE.

        Args:
            z: latent code
        Returns:
            log probability density
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def add_argparse_args(parent_parser):
        """
        Add the necessary arguments for instantiation of this SDE class to an argparse ArgumentParser.
        """
        pass

    def discretize(self, x, t, *args):
        """Discretize the SDE in the form: x_{i+1} = x_i + f_i(x_i) + G_i z_i.

        Useful for reverse diffusion sampling and probabiliy flow sampling.
        Defaults to Euler-Maruyama discretization.

        Args:
            x: a torch tensor
            t: a torch float representing the time step (from 0 to `self.T`)

        Returns:
            f, G
        """
        dt = 1 / self.N
        drift, diffusion = self.sde(x, t, *args)
        f = drift * dt
        G = diffusion * torch.sqrt(torch.tensor(dt, device=t.device))
        return f, G

    def reverse(oself, score_model, probability_flow=False, diffusion_power_gradient=None):
        """Create the reverse-time SDE/ODE.

        Args:
            score_model: A function that takes x, t and y and returns the score.
            probability_flow: If `True`, create the reverse-time ODE used for probability flow sampling.
            diffusion_power_gradient: func or None. if the diffusion function has a gradient with respect to the process X, we need to include it in the reverse SDE
            (cf Anderson1982 or Appendix A of Song2021)
        """
        N = oself.N
        T = oself.T
        sde_fn = oself.sde
        discretize_fn = oself.discretize
        std_fn = oself._std

        # Build the class for reverse-time SDE.
        class RSDE(oself.__class__):
            def __init__(self):
                self.N = N
                self.probability_flow = probability_flow
                self.diffusion_power_gradient = diffusion_power_gradient

            @property
            def T(self):
                return T

            def sde(self, x, t, conditioning, sde_input, **kwargs):
                """Create the drift and diffusion functions for the reverse SDE/ODE."""
                rsde_parts = self.rsde_parts(x, t, conditioning, sde_input, **kwargs)
                total_drift, diffusion = rsde_parts["total_drift"], rsde_parts["diffusion"]
                score = rsde_parts["score"]
                return total_drift, diffusion, score

            def rsde_parts(self, x, t, conditioning, sde_input, **kwargs):
                sde_drift, sde_diffusion = sde_fn(x, t, sde_input)
                score = score_model(x, t, score_conditioning=conditioning)
                if sde_diffusion.ndim < x.ndim:
                    sde_diffusion = sde_diffusion.view(*sde_diffusion.size(), *((1,)*(x.ndim - sde_diffusion.ndim)))
                score_drift = sde_diffusion**2 * score * (0.5 if self.probability_flow else 1.)
                diffusion = torch.zeros_like(sde_diffusion) if self.probability_flow else sde_diffusion
                total_drift = sde_drift - score_drift
                return {
                    'total_drift': total_drift, 'diffusion': diffusion, 'sde_drift': sde_drift,
                    'sde_diffusion': sde_diffusion, 'score_drift': score_drift, 'score': score
                }

            def discretize(self, x, t, conditioning, sde_input, **kwargs):
                """Create discretized iteration rules for the reverse diffusion sampler."""
                f, G = discretize_fn(x, t, sde_input)
                if G.ndim < x.ndim:
                    G = G.view(*G.size(), *((1,)*(x.ndim - G.ndim)))
                score = score_model(x, t, score_conditioning=conditioning)
                rev_f = f - G**2 * score * (0.5 if self.probability_flow else 1.)
                rev_G = torch.zeros_like(G) if self.probability_flow else G
                return rev_f, rev_G, score

        return RSDE()

    @abc.abstractmethod
    def copy(self):
        pass


@SDERegistry.register("ve")
class VESDE(SDE):
    def __init__(self, sigma_min, sigma_max, N=1000, **ignored_kwargs):
        """Construct a Variance Exploding SDE.

        dx = sigma(t) dw

        with

        sigma(t) = sigma_min (sigma_max/sigma_min)^t * sqrt(2 log(sigma_max/sigma_min))

        Args:
            sigma_min: smallest sigma.
            sigma_max: largest sigma.
            N: number of discretization steps
        """
        super().__init__(N)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.logsig = np.log(self.sigma_max / self.sigma_min)
        self.N = N

    def copy(self):
        return VESDE(self.sigma_min, self.sigma_max, N=self.N)

    @property
    def T(self):
        return 1

    def sde(self, x, t, *args, **kwargs):
        sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        diffusion = sigma * np.sqrt(2 * self.logsig)
        return .0, diffusion

    def _mean(self, x0, t, *args, **kwargs):
        return x0

    def _std(self, t, *args, **kwargs):
        # This is a full solution to the ODE for P(t) in our derivations, after choosing g(s) as in self.sde()
        sigma_min, logsig = self.sigma_min, self.logsig
        return sigma_min*torch.sqrt(torch.exp(2 * logsig * t) - 1)

    def _inverse_std(self, sigma, *args, **kwargs):
        sigma_min, logsig = self.sigma_min, self.logsig
        return torch.log(sigma**2/sigma_min**2 + 1)/(2*logsig)

    def marginal_prob(self, x0, t, *args, **kwargs):
        return self._mean(x0, t), self._std(t)

    def prior_sampling(self, shape, y, unconditional_prior=True, **kwargs):
        if shape != y.shape:
            warnings.warn(f"Target shape {shape} does not match shape of y {y.shape}! Ignoring target shape.")
        std = self._std(torch.ones((y.shape[0],), device=y.device))
        std = std.view(std.size(0), *(1,)*(y.ndim - std.ndim))
        if unconditional_prior:
            return torch.randn_like(y) * std
        else:
            return y + torch.randn_like(y) * std

    def prior_logp(self, z):
        raise NotImplementedError("prior_logp for VE SDE not yet implemented!")

    def tweedie_from_score(self, score, x, t, *args):
        sigma = self._std(t)
        sigma = sigma.view(sigma.size(0), *(1,)*(score.ndim - sigma.ndim))
        return x + sigma**2 * score
    
    def score_from_tweedie(self, tweedie, x, t, *args):
        sigma = self._std(t)
        sigma = sigma.view(sigma.size(0), *(1,)*(tweedie.ndim - sigma.ndim))
        return 1 / sigma**2 * (tweedie - x)

    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--sde_n", type=int, default=1000,
            help="The number of timesteps in the SDE discretization. 1000 by default")
        parser.add_argument("--sigma_min", type=float, default=0.05, 
            help="The minimum sigma to use.")
        parser.add_argument("--sigma_max", type=float, default=0.5, 
            help="The maximum sigma to use.")
        return parser




@SDERegistry.register("ve-karras")
class VEKarrasSDE(SDE):
    def __init__(self, N=1000, sigma_min=0.01, sigma_max=0.5, rho=7, **ignored_kwargs):
        """Construct a Variance Exploding SDE.

        dx = sqrt( 2 sigma(t) ) dw

        with

        sigma(t) = t

        """
        super().__init__(N)
        self.N = N
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho

    def copy(self):
        return VEKarrasSDE(N=self.N)

    @property
    def T(self):
        return self.sigma_max

    def sde(self, x, t, *args, **kwargs):
        diffusion = torch.sqrt(2 * t)
        return .0, diffusion

    def _mean(self, x0, t, *args, **kwargs):
        return x0

    def _std(self, t, *args, **kwargs):
        sigma = t
        return sigma

    def _inverse_std(self, sigma, *args, **kwargs):
        t = sigma
        return t

    def marginal_prob(self, x0, t, *args, **kwargs):
        return self._mean(x0, t), self._std(t)

    def prior_sampling(self, shape, y, unconditional_prior=True, **kwargs):
        if shape != y.shape:
            warnings.warn(f"Target shape {shape} does not match shape of y {y.shape}! Ignoring target shape.")
        std = self._std(self.T * torch.ones((y.shape[0],), device=y.device))
        std = std.view(std.size(0), *(1,)*(y.ndim - std.ndim))
        if unconditional_prior:
            return torch.randn_like(y) * std
        else:
            return y + torch.randn_like(y) * std

    def prior_logp(self, z):
        raise NotImplementedError("prior_logp for VE SDE not yet implemented!")

    def tweedie_from_score(self, score, x, t, *args):
        sigma = self._std(t)
        sigma = sigma.view(sigma.size(0), *(1,)*(score.ndim - sigma.ndim))
        return x + sigma**2 * score
    
    def score_from_tweedie(self, tweedie, x, t, *args):
        sigma = self._std(t)
        sigma = sigma.view(sigma.size(0), *(1,)*(tweedie.ndim - sigma.ndim))
        return 1 / sigma**2 * (tweedie - x)

    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--sde_n", type=int, default=1000,
            help="The number of timesteps in the SDE discretization. 1000 by default")
        parser.add_argument("--sigma_min", type=float, default=0.01, 
            help="The minimum sigma to use.")
        parser.add_argument("--sigma_max", type=float, default=0.5, 
            help="The maximum sigma to use.")
        parser.add_argument("--rho", type=float, default=7.)
        return parser
