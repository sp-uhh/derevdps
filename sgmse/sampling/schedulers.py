import abc

import torch
import numpy as np

from sgmse.util.registry import Registry


SchedulerRegistry = Registry("Scheduler")


class Scheduler(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, N, **kwargs):
        super().__init__()
        self.N = N

    def reverse_timesteps(self, T):
        lin_timesteps = torch.linspace(T, -1/self.N, self.N)
        timesteps = self.continuous_step(lin_timesteps)
        return timesteps

    @abc.abstractmethod
    def continuous_step(self, a):
        pass

@SchedulerRegistry.register("ve-song")
class VESongScheduler(Scheduler):

    def __init__(self, N, eps=0, sigma_min=5e-2, sigma_max=5e-1, **kwargs):
        super().__init__(N)
        self.eps = eps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def continuous_step(self, a):
        return self.sigma_min**2 * (self.sigma_max / self.sigma_min)**(2*a)
    
@SchedulerRegistry.register("edm")
class EDMScheduler(Scheduler):

    def __init__(self, N, eps=0, sigma_min=1e-5, sigma_max=150., rho=7, **kwargs):
        super().__init__(N)
        self.eps = eps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho

    def continuous_step(self, a):
        return (self.sigma_min**(1/self.rho) + a * (self.sigma_max**(1/self.rho) - self.sigma_min**(1/self.rho))) **self.rho