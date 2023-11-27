import abc

import torch
import numpy as np

from sgmse.util.registry import Registry


PredictorRegistry = Registry("Predictor")


class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, sde, score_fn, probability_flow):
        super().__init__()
        self.sde = sde
        self.rsde = sde.reverse(score_fn, probability_flow=probability_flow)
        self.score_fn = score_fn
        self.probability_flow = probability_flow

    @abc.abstractmethod
    def update_fn(self, x, timesteps, i, *args):
        """One update of the predictor.

        Args:
            x: A PyTorch tensor representing the current state
            t: A Pytorch tensor representing the current time step.
            *args: Possibly additional arguments, in particular `y` for OU processes

        Returns:
            x: A PyTorch tensor of the next state.
            x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
        """
        pass

    def debug_update_fn(self, x, t, *args):
        raise NotImplementedError(f"Debug update function not implemented for predictor {self}.")

    
@PredictorRegistry.register('euler-maruyama')
class EulerMaruyamaPredictor(Predictor):
    """
    1st-order method for solving SDEs, classical Euler-Maruyama scheme
    """
    def __init__(self, sde, score_fn, probability_flow):
        super().__init__(sde, score_fn, probability_flow=probability_flow)

    def update_fn(self, x, t, dt, conditioning, sde_input, **kwargs):
        z = torch.randn_like(x)
        f, g, score = self.rsde.sde(x, t, conditioning, sde_input, **kwargs)
        x_mean = x + f * dt
        if g.ndim < x.ndim:
            g = g.view( *g.size(), *((1,)*(x.ndim - g.ndim)) )
        x = x_mean + g * torch.sqrt(-dt) * z
        return x, x_mean, score

@PredictorRegistry.register('euler-heun')
class EulerHeunPredictor(Predictor):
    """
    2nd-order method for solving ODEs, classical trapezoidal scheme (can be seen as Euler + a 2nd-order corrector using implicit Euler)
    """
    def __init__(self, sde, score_fn, probability_flow):
        super().__init__(sde, score_fn, probability_flow=probability_flow)

    def update_fn(self, x, t, dt, conditioning, sde_input, **kwargs):

        f, _, score = self.rsde.sde(x, t, conditioning, sde_input, **kwargs)
        x_mean = x + f * dt
        if self.sde._std(t + dt) > 0:
            f_next, _, score = self.rsde.sde(x_mean, t+dt, conditioning, sde_input, probability_flow=self.probability_flow, **kwargs)
            x_mean = x + .5 * (f + f_next) * dt
        x = x_mean
        return x, x_mean, score

@PredictorRegistry.register('none')
class NonePredictor(Predictor):
    """An empty predictor that does nothing."""

    def __init__(self, *args, **kwargs):
        pass

    def update_fn(self, x, t, *args, **kwargs):
        return x, x


@PredictorRegistry.register('euler-heun-dps')
class EulerHeunDPSPredictor(Predictor):
    """
    2nd-order method for solving ODEs, classical trapezoidal scheme (can be seen as Euler + a 2nd-order corrector using implicit Euler)
    Merging here the DPS posterior, as this one performs DPS twice for each step: once with the score obtained from the point at t=t and
    once for the midpoint t=t+dt/2
    This is of course quite expensive compared to classical Euler DPS, but works quite well
    Makes code messy though, that is unfortunate.
    """
    def __init__(self, sde, score_fn,  operator, linearization, zeta, probability_flow=False,):
        super().__init__(sde, score_fn, probability_flow=probability_flow)
        self.operator = operator
        self.linearization = linearization
        self.zeta = zeta

    def get_likelihood_score(self, score, x, t, measurement, A):

        x_0_hat=self.tweedie_from_score(score, x, t)

        measurement_linear, x_0_hat_linear = self.linearization(measurement.squeeze(0)).unsqueeze(0), self.linearization(x_0_hat.squeeze(0)).unsqueeze(0)
        self.operator.load_weights(A.squeeze(0))
        measurement_estimated = self.operator.forward(x_0_hat_linear)
        difference = measurement_linear - measurement_estimated
        norm = torch.linalg.norm(difference)
        norm_grad = torch.autograd.grad(outputs=norm, inputs=x)[0]
        normguide = torch.linalg.norm(norm_grad)/x.shape[-1]**0.5

        zeta_t = self.zeta/(normguide+1e-6)

        return -zeta_t*norm_grad/t, norm

    def update_fn(self, x, t, dt, conditioning, sde_input, measurement, A, **kwargs):

        _, _, score = self.rsde.sde(x, t, conditioning, sde_input, **kwargs)
        likelihood_score, distance = self.get_likelihood_score(score, x,t, measurement, A)

        f = -t.view(*t.size(), *((1,)*(score.ndim-t.ndim)))*(score+likelihood_score)
        x_mean = x + f * dt 
        if (self.sde._std(t + dt) > 0).any():
            t_next = t + dt
            f_next, _, score_next = self.rsde.sde(x_mean, t_next, conditioning, sde_input, **kwargs)
            likelihood_score_next, distance = self.get_likelihood_score(score_next, x_mean,t_next, measurement, A)
            f_next = -t_next.view(*t_next.size(), *((1,)*(score_next.ndim-t_next.ndim)))*(score_next+likelihood_score_next)
            x_mean = x + .5 * (f + f_next) * dt
        x = x_mean
        return x, x_mean, score, A, distance

    def tweedie_from_score(self, score, x, t, *args):
        return self.sde.tweedie_from_score(score, x, t, *args)

    def grad_required(self, t, **kwargs):
        return True