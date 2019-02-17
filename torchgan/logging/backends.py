import os
import subprocess

from ..utils import getenv_defaults

# Backends available for Visualization

# Tensorboard
TENSORBOARD_LOGGING = int(
    os.getenv("TENSORBOARD_LOGGING", getenv_defaults("tensorboardX"))
)
if TENSORBOARD_LOGGING == 1 and getenv_defaults("tensorboardX") == 0:
    raise Exception(
        "TensorboardX is not installed. Install it or set TENSORBOARD_LOGGING to 0"
    )

# Console
CONSOLE_LOGGING = int(os.getenv("CONSOLE_LOGGING", 1))

# Visdom
VISDOM_LOGGING = int(os.getenv("VISDOM_LOGGING", getenv_defaults("visdom")))
if VISDOM_LOGGING == 1:
    if getenv_defaults("visdom") == 0:
        raise Exception(
            "Visdom is not installed. Install it or set VISDOM_LOGGING to 0"
        )
