import torch
from pkgutil import iter_modules

def reduce(x, reduction=None):
    if reduction == "elementwise_mean":
        return torch.mean(x)
    elif reduction == "sum":
        return torch.sum(x)
    else:
        return x

def getenv_defaults(module_name):
    return int(module_name in (name for loader, name, ispkg in iter_modules()))
