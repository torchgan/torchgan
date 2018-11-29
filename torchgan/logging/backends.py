from ..utils import getenv_defaults
import os

# Backends available for Visualization

# Tensorboard
TENSORBOARD_LOGGING = os.getenv("TENSORBOARD_LOGGING", getenv_defaults("tensorboardX"))
if TENSORBOARD_LOGGING == 1 and getenv_defaults("tensorboardX") == 0:
    raise Exception("TensorboardX is not installed. Install it or set TENSORBOARD_LOGGING to 0")

# Console
CONSOLE_LOGGING = os.getenv("CONSOLE_LOGGING", 1)

# Matplotlib
MATPLOTLIB_LOGGING = os.getenv("MATPLOTLIB_LOGGING", getenv_defaults("matplotlib"))
if MATPLOTLIB_LOGGING == 1 and getenv_defaults("matplotlib") == 0:
    raise Exception("Matplotlib is not installed. Install it or set MATPLOTLIB_LOGGING to 0")

# Seaborn
SEABORN_LOGGING = os.getenv("SEABORN_LOGGING", getenv_defaults("seaborn"))
if SEABORN_LOGGING != os.getenv("SEABORN_LOGGING") and SEABORN_LOGGING == 1 and MATPLOTLIB_LOGGING == 1:
    SEABORN_LOGGING = 0
if SEABORN_LOGGING == 1 and getenv_defaults("seaborn") == 0:
    raise Exception("Seaborn is not installed. Install it or set SEABORN_LOGGING to 0")

if SEABORN_LOGGING == 1 and MATPLOTLIB_LOGGING == 1:
    raise Exception("Both matplotlib and seaborn can't be used together. Please turn off one logging")
