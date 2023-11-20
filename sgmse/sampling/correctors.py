import abc
import torch

from sgmse import sdes
from sgmse.util.registry import Registry


CorrectorRegistry = Registry("Corrector")


class Corrector(abc.ABC):
    """The abstract class for a corrector algorithm."""

    def __init__(self, sde, score_fn, probability_flow, r, n_steps):
        super().__init__()
        self.rsde = sde.reverse(score_fn, probability_flow=probability_flow)
        self.score_fn = score_fn
        self.r = r
        self.n_steps = n_steps

    @abc.abstractmethod
    def update_fn(self, x, timesteps, i, *args):
        """One update of the corrector.

        Args:
            x: A PyTorch tensor representing the current state
            timesteps: A PyTorch tensor representing the time steps.
            i: int pointing to the current timestep index
            *args: Possibly additional arguments, in particular `y` for OU processes

        Returns:
            x: A PyTorch tensor of the next state.
            x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
        """
        pass


@CorrectorRegistry.register(name='ald')
class AnnealedLangevinDynamics(Corrector):
    """The original annealed Langevin dynamics predictor in NCSN/NCSNv2."""
    def __init__(self, sde, score_fn, probability_flow, r, n_steps):
        super().__init__(sde, score_fn, probability_flow, r, n_steps)
        self.sde = sde
        self.score_fn = score_fn
        self.r = r
        self.n_steps = n_steps

    def update_fn(self, x, t, dt, conditioning, sde_input, **kwargs):
        n_steps = self.n_steps
        target_r = self.r
        std = self.sde.marginal_prob(x, t, sde_input)[1]

        for _ in range(n_steps):
            grad = self.score_fn(x, t, score_conditioning=conditioning)
            noise = torch.randn_like(x)
            step_size = (target_r * std) ** 2 * 2
            if step_size.ndim < x.ndim:
                step_size = step_size.view( *step_size.size(), *((1,)*(x.ndim - step_size.ndim)) )
            x_mean = x + step_size * grad
            x = x_mean + noise * torch.sqrt(step_size * 2)

        return x, x_mean


@CorrectorRegistry.register(name='none')
class NoneCorrector(Corrector):
    """An empty corrector that does nothing."""

    def __init__(self, *args, **kwargs):
        self.r = 0
        self.n_steps = 0
        pass

    def update_fn(self, x, timesteps, i, *args, **kwargs):
        return x, x
