import abc

import torch
import numpy as np

def get_optimizer(optimizer_name, params, lr):
    if optimizer_name == "adam":
        return torch.optim.Adam(params, lr=lr, betas=(0.9, 0.99), weight_decay=0.0)
    elif optimizer_name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    else:
        raise NotImplementedError