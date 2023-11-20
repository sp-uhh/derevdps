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
        print("pred f rev", torch.linalg.norm(f))
        print("pred g rev", torch.linalg.norm(g))
        print("pred score", torch.linalg.norm(score))
        print("pred dt", dt)
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

        f, g, score = self.rsde.sde(x, t, conditioning, sde_input, **kwargs)
        x_mean = x + f * dt
        if self.sde._std(t + dt) > 0:
            f_next, _, score_next = self.rsde.sde(x_mean, t+dt, conditioning, sde_input, probability_flow=self.probability_flow, **kwargs)
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
