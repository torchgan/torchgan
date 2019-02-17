from pkgutil import iter_modules

import torch


def reduce(x, reduction=None):
    r"""Applies reduction on a torch.Tensor.

    Args:
        x (torch.Tensor): The tensor on which reduction is to be applied.
        reduction (str, optional): The reduction to be applied. If ``mean`` the  mean value of the
            Tensor is returned. If ``sum`` the elements of the Tensor will be summed. If none of the
            above then the Tensor is returning without any change.

    Returns:
        As per the above ``reduction`` convention.
    """
    if reduction == "mean":
        return torch.mean(x)
    elif reduction == "sum":
        return torch.sum(x)
    else:
        return x


def getenv_defaults(module_name):
    r"""Determines if a particular package is installed in the system.

    Args:
        module_name (str): The name of the package to be found.

    Returns:
        1 if package is installed else 0
    """
    return int(module_name in (name for loader, name, ispkg in iter_modules()))
