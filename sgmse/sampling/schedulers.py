import abc

import torch
import numpy as np

from sgmse.util.registry import Registry


SchedulerRegistry = Registry("Scheduler")


class Scheduler(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, N, eps, **kwargs):
        super().__init__()
        self.N = N
        self.eps = eps

    @abc.abstractmethod
    def timesteps(self):
        pass

@SchedulerRegistry.register("linear")
class LinearScheduler(Scheduler):
    
    def timesteps(self):
        timesteps = torch.linspace(1., self.eps, self.N)
        return torch.cat([timesteps, torch.Tensor([0.])])

@SchedulerRegistry.register("karras")
class KarrasScheduler(Scheduler):

    def __init__(self, N, eps, sigma_min=1e-5, sigma_max=150., rho=7, **kwargs):
        super().__init__(N, eps)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho

    def timesteps(self):
        lin_timesteps = torch.linspace(self.eps, 1., self.N)
        timesteps = (self.sigma_max**(1/self.rho) + lin_timesteps * (self.sigma_min**(1/self.rho) - self.sigma_max**(1/self.rho))) **self.rho
        return torch.cat([timesteps, torch.Tensor([0.])])