import abc

import torch
import numpy as np

from sgmse.util.registry import Registry


SchedulerRegistry = Registry("Scheduler")


class Scheduler(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, N, eps=1e-6, **kwargs):
        super().__init__()
        self.N = N
        self.eps = eps

    def reverse_timesteps(self, T):
        lin_timesteps = torch.linspace(T, self.eps, self.N)
        # lin_timesteps = torch.linspace(T, 0., self.N)
        timesteps = self.continuous_step(lin_timesteps)
        return torch.cat([timesteps, torch.Tensor([0.])])

    @abc.abstractmethod
    def continuous_step(self, a):
        pass

@SchedulerRegistry.registesr('linear')
class LinearScheduler(Scheduler):

    def __init__(self, N, eps=1e-3, **kwargs):
        super().__init__(N, eps)

    def continuous_step(self, a):
        return a
    
@SchedulerRegistry.register("ve")
class VESongScheduler(Scheduler):

    def __init__(self, N, eps=3e-2, sigma_min=5e-2, sigma_max=5e-1, **kwargs):
        super().__init__(N, eps)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def continuous_step(self, a):
        return self.sigma_min**2 * (self.sigma_max / self.sigma_min)**(2*a)
    
@SchedulerRegistry.register("edm")
class EDMScheduler(Scheduler):

    def __init__(self, N, eps=1e-6, sigma_min=1e-5, sigma_max=150., rho=7, **kwargs):
        print(eps)
        super().__init__(N, eps)

        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho

    def continuous_step(self, a):
        return (self.sigma_min**(1/self.rho) + a * (self.sigma_max**(1/self.rho) - self.sigma_min**(1/self.rho))) **self.rho