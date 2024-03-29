import abc

import torch
import numpy as np

from sgmse.util.registry import Registry

PosteriorRegistry = Registry("Posterior")

class Posterior(abc.ABC):

    def __init__(self, sde, operator, linearization, zeta):
        super().__init__()
        self.linearization = linearization
        self.sde = sde
        self.operator = operator
        self.zeta = zeta

    @abc.abstractmethod
    def update_fn(self, x, t, y, score, *args):
        """One update of the measurement conditioner.

        Args:
            x: A PyTorch tensor representing the current state given by the predictor
            t: A Pytorch tensor representing the current time step.
            y: A PyTorch tensor representing the measurement
            score: The score obtained at time t
            *args: Possibly additional arguments, in particular A the known/estimated operator

        Returns:
            x: A PyTorch tensor of the next state.
            A: The updated estimator (if in a blind setting, otherwise just copy)
        """
        pass

    @abc.abstractmethod
    def grad_required(self, t):
        pass

    def tweedie_from_score(self, score, x, t, *args):
        return self.sde.tweedie_from_score(score, x, t, *args)

    def project(self, data, noisy_measurement, **kwargs):
        return self.operator.project(data=data, measurement=noisy_measurement, **kwargs)
    

@PosteriorRegistry.register('none')
class NoPosteriorSampling(Posterior):
    
    def update_fn(self, x, *args, **kwargs):
        return x, None, torch.Tensor([0.]), None, None

    def grad_required(self, t, **kwargs):
        return False

@PosteriorRegistry.register('dps')
class PosteriorSampling(Posterior):
    
    # def update_fn(self, x, t, dt, measurement, sde_input, score, A, *args, **kwargs):
    def update_fn(self, x, x_grad_ref, t, dt, measurement, sde_input, score, A, *args, **kwargs):
        # x_0_hat = self.tweedie_from_score(score, x, t, sde_input)
        x_0_hat = self.tweedie_from_score(score, x_grad_ref, t, sde_input)

        measurement_linear, x_0_hat_linear = self.linearization(measurement.squeeze(0)).unsqueeze(0), self.linearization(x_0_hat.squeeze(0)).unsqueeze(0)
        self.operator.load_weights(A.squeeze(0))
        measurement_estimated = self.operator.forward(x_0_hat_linear)
        difference = measurement_linear - measurement_estimated
        norm = torch.linalg.norm(difference)

        # norm_grad = torch.autograd.grad(outputs=norm, inputs=x)[0]
        norm_grad = torch.autograd.grad(outputs=norm, inputs=x_grad_ref)[0]
        normguide = torch.linalg.norm(norm_grad)/x.shape[-1]**0.5 + 1e-6

        x  = x + norm_grad * self.zeta / normguide * dt #dt < 0
        return x, A, norm, measurement_estimated, x_0_hat_linear

    def grad_required(self, t, **kwargs):
        return True
        # return self.zeta > 0
        
@PosteriorRegistry.register('state-dps')
class StateDPSPosteriorSampling(Posterior):
    
    def update_fn(self, x, t, dt, measurement, sde_input, score, A, *args, **kwargs):
        x = x.requires_grad_(True)
        measurement_linear, x_linear = self.linearization(measurement.squeeze(0)).unsqueeze(0), self.linearization(x.squeeze(0)).unsqueeze(0)
        self.operator.load_weights(A)

        measurement_estimated = self.operator.forward(x_linear.squeeze(0)).unsqueeze(0)
        difference = measurement_linear - measurement_estimated
        norm = torch.linalg.norm(difference)
        norm_grad = torch.autograd.grad(outputs=norm, inputs=x)[0]
        normguide = torch.linalg.norm(norm_grad)/x.shape[-1]**0.5 + 1e-6

        x  = x + norm_grad * self.zeta / normguide * dt #dt < 0
        return x, A, norm, measurement_estimated, x_linear

    def grad_required(self, t, **kwargs):
        return self.zeta > 0