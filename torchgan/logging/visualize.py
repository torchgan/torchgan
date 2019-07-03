import torch
import torchvision

from ..models.model import Discriminator, Generator
from .backends import *

if TENSORBOARD_LOGGING == 1:
    from tensorboardX import SummaryWriter
if VISDOM_LOGGING == 1:
    import visdom

__all__ = [
    "Visualize",
    "LossVisualize",
    "MetricVisualize",
    "GradientVisualize",
    "ImageVisualize",
]


class Visualize(object):
    r"""Base class for all Visualizations.

    Args:
        visualize_list (list, optional): List of the functions needed for visualization.
        visdom_port (int, optional): Port to log using ``visdom``. The visdom server needs to be
            manually started at this port else an error will be thrown and the code will crash.
            This is ignored if ``VISDOM_LOGGING`` is ``0``.
        log_dir (str, optional): Directory where TensorboardX should store the logs. This is
            ignored if ``TENSORBOARD_LOGGING`` is ``0``.
        writer (tensorboardX.SummaryWriter, optonal): Send a `SummaryWriter` if you
            don't want to start a new SummaryWriter.
    """

    def __init__(self, visualize_list, visdom_port=8097, log_dir=None, writer=None):
        self.logs = {}
        for item in visualize_list:
            name = type(item).__name__
            self.logs[name] = []
        self.step = 1
        if TENSORBOARD_LOGGING == 1:
            self._build_tensorboard(log_dir, writer)
        if VISDOM_LOGGING == 1:
            self._build_visdom(visdom_port)

    def _build_tensorboard(self, log_dir, writer):
        r"""Starts the tensorboard logging utilities.

        Args:
            log_dir (str, optional): Directory where TensorboardX should store the logs.
            writer (tensorboardX.SummaryWriter, optonal): Send a `SummaryWriter` if you
                don't want to start a new SummaryWriter.
        """
        self.writer = SummaryWriter(log_dir) if writer is None else writer

    def _build_visdom(self, port):
        r"""Starts the visdom logging utilities.

        Args:
            port (int, optional): Port to log using ``visdom``. A deafult server is started at port
                ``8097``. So manually a new server has to be started if the post is changed.
        """
        self.vis = visdom.Visdom(port=port)

    def step_update(self):
        r"""Helper function which updates the step at the end of
        one print iteration.
        """
        self.step += 1

    def log_tensorboard(self):
        r"""Tensorboard logging function. Needs to be defined in the subclass

        :raises NotImplementedError:
        """
        raise NotImplementedError

    def log_console(self):
        r"""Console logging function. Needs to be defined in the subclass

        :raises NotImplementedError:
        """
        raise NotImplementedError

    def log_visdom(self):
        r"""Visdom logging function. Needs to be defined in the subclass

        :raises NotImplementedError:
        """
        raise NotImplementedError

    def __call__(
        self,
        *args,
        lock_console=False,
        lock_tensorboard=False,
        lock_visdom=False,
        **kwargs
    ):
        if not lock_console and CONSOLE_LOGGING == 1:
            self.log_console(*args, **kwargs)
        if not lock_tensorboard and TENSORBOARD_LOGGING == 1:
            self.log_tensorboard(*args, **kwargs)
        if not lock_visdom and VISDOM_LOGGING == 1:
            self.log_visdom(*args, **kwargs)
        self.step_update()


class LossVisualize(Visualize):
    r"""This class provides the Visualizations for Generator and Discriminator Losses.

    Args:
        visualize_list (list, optional): List of the functions needed for visualization.
        visdom_port (int, optional): Port to log using ``visdom``. The visdom server needs to be
            manually started at this port else an error will be thrown and the code will crash.
            This is ignored if ``VISDOM_LOGGING`` is ``0``.
        log_dir (str, optional): Directory where TensorboardX should store the logs. This is
            ignored if ``TENSORBOARD_LOGGING`` is ``0``.
        writer (tensorboardX.SummaryWriter, optonal): Send a `SummaryWriter` if you
            don't want to start a new SummaryWriter.
    """

    def log_tensorboard(self, running_losses):
        r"""Tensorboard logging function. This function logs the following:

        - ``Running Discriminator Loss``
        - ``Running Generator Loss``
        - ``Running Losses``
        - Loss Values of the individual Losses.

        Args:
            running_losses (dict): A dict with 2 items namely, ``Running Discriminator Loss``,
                and ``Running Generator Loss``.
        """
        self.writer.add_scalar(
            "Running Discriminator Loss",
            running_losses["Running Discriminator Loss"],
            self.step,
        )
        self.writer.add_scalar(
            "Running Generator Loss",
            running_losses["Running Generator Loss"],
            self.step,
        )
        self.writer.add_scalars("Running Losses", running_losses, self.step)
        for name, value in self.logs.items():
            val = value[-1]
            if type(val) is tuple:
                self.writer.add_scalar(
                    "Losses/{}-Generator".format(name), val[0], self.step
                )
                self.writer.add_scalar(
                    "Losses/{}-Discriminator".format(name), val[1], self.step
                )
            else:
                self.writer.add_scalar("Losses/{}".format(name), val, self.step)

    def log_console(self, running_losses):
        r"""Console logging function. This function logs the mean ``generator`` and ``discriminator``
        losses.

        Args:
            running_losses (dict): A dict with 2 items namely, ``Running Discriminator Loss``,
                and ``Running Generator Loss``.
        """
        for name, val in running_losses.items():
            print("Mean {} : {}".format(name, val))

    def log_visdom(self, running_losses):
        r"""Visdom logging function. This function logs the following:

        - ``Running Discriminator Loss``
        - ``Running Generator Loss``
        - ``Running Losses``
        - Loss Values of the individual Losses.

        Args:
            running_losses (dict): A dict with 2 items namely, ``Running Discriminator Loss``,
                and ``Running Generator Loss``.
        """
        self.vis.line(
            [running_losses["Running Discriminator Loss"]],
            [self.step],
            win="Running Discriminator Loss",
            update="append",
            opts=dict(
                title="Running Discriminator Loss",
                xlabel="Time Step",
                ylabel="Running Loss",
            ),
        )
        self.vis.line(
            [running_losses["Running Generator Loss"]],
            [self.step],
            win="Running Generator Loss",
            update="append",
            opts=dict(
                title="Running Generator Loss",
                xlabel="Time Step",
                ylabel="Running Loss",
            ),
        )
        self.vis.line(
            [
                [
                    running_losses["Running Discriminator Loss"],
                    running_losses["Running Generator Loss"],
                ]
            ],
            [self.step],
            win="Running Losses",
            update="append",
            opts=dict(
                title="Running Losses",
                xlabel="Time Step",
                ylabel="Running Loss",
                legend=["Discriminator", "Generator"],
            ),
        )
        for name, value in self.logs.items():
            val = value[-1]
            if type(val) is tuple:
                name1 = "{}-Generator".format(name)
                name2 = "{}-Discriminator".format(name)
                self.vis.line(
                    [val[0]],
                    [self.step],
                    win=name1,
                    update="append",
                    opts=dict(title=name1, xlabel="Time Step", ylabel="Loss Value"),
                )
                self.vis.line(
                    [val[1]],
                    [self.step],
                    win=name2,
                    update="append",
                    opts=dict(title=name2, xlabel="Time Step", ylabel="Loss Value"),
                )
            else:
                self.vis.line(
                    [val],
                    [self.step],
                    win=name,
                    update="append",
                    opts=dict(title=name, xlabel="Time Step", ylabel="Loss Value"),
                )

    def __call__(self, trainer, **kwargs):
        running_generator_loss = (
            trainer.loss_information["generator_losses"]
            / trainer.loss_information["generator_iters"]
        )
        running_discriminator_loss = (
            trainer.loss_information["discriminator_losses"]
            / trainer.loss_information["discriminator_iters"]
        )
        running_losses = {
            "Running Discriminator Loss": running_discriminator_loss,
            "Running Generator Loss": running_generator_loss,
        }
        super(LossVisualize, self).__call__(running_losses, **kwargs)


class MetricVisualize(Visualize):
    r"""This class provides the Visualizations for Metrics.

    Args:
        visualize_list (list, optional): List of the functions needed for visualization.
        visdom_port (int, optional): Port to log using ``visdom``. The visdom server needs to be
            manually started at this port else an error will be thrown and the code will crash.
            This is ignored if ``VISDOM_LOGGING`` is ``0``.
        log_dir (str, optional): Directory where TensorboardX should store the logs. This is
            ignored if ``TENSORBOARD_LOGGING`` is ``0``.
        writer (tensorboardX.SummaryWriter, optonal): Send a `SummaryWriter` if you
            don't want to start a new SummaryWriter.
    """

    def log_tensorboard(self):
        r"""Tensorboard logging function. This function logs the values of the individual metrics.
        """
        for name, value in self.logs.items():
            self.writer.add_scalar("Metrics/{}".format(name), value[-1], self.step)

    def log_console(self):
        r"""Console logging function. This function logs the mean metrics.
        """
        for name, val in self.logs.items():
            print("{} : {}".format(name, val[-1]))

    def log_visdom(self):
        r"""Visdom logging function. This function logs the values of the individual metrics.
        """
        for name, value in self.logs.items():
            self.vis.line(
                [value[-1]],
                [self.step],
                win=name,
                update="append",
                opts=dict(title=name, xlabel="Time Step", ylabel="Metric Value"),
            )


class GradientVisualize(Visualize):
    r"""This class provides the Visualizations for the Gradients.

    Args:
        visualize_list (list, optional): List of the functions needed for visualization.
        visdom_port (int, optional): Port to log using ``visdom``. The visdom server needs to be
            manually started at this port else an error will be thrown and the code will crash.
            This is ignored if ``VISDOM_LOGGING`` is ``0``.
        log_dir (str, optional): Directory where TensorboardX should store the logs. This is
            ignored if ``TENSORBOARD_LOGGING`` is ``0``.
        writer (tensorboardX.SummaryWriter, optonal): Send a `SummaryWriter` if you
            don't want to start a new SummaryWriter.
    """

    def __init__(self, visualize_list, visdom_port=8097, log_dir=None, writer=None):
        if visualize_list is None or len(visualize_list) == 0:
            raise Exception("Gradient Visualizer requires list of model names")
        self.logs = {}
        for item in visualize_list:
            self.logs[item] = [0.0]
        self.step = 1
        if TENSORBOARD_LOGGING == 1:
            self._build_tensorboard(log_dir, writer)
        if VISDOM_LOGGING == 1:
            self._build_visdom(visdom_port)

    def log_tensorboard(self, name):
        r"""Tensorboard logging function. This function logs the values of the individual gradients.

        Args:
            name (str): Name of the model whose gradients are to be logged.
        """
        self.writer.add_scalar(
            "Gradients/{}".format(name),
            self.logs[name][len(self.logs[name]) - 1],
            self.step,
        )

    def log_console(self, name):
        r"""Console logging function. This function logs the mean gradients.

        Args:
            name (str): Name of the model whose gradients are to be logged.
        """
        print(
            "{} Gradients : {}".format(name, self.logs[name][len(self.logs[name]) - 1])
        )

    def log_visdom(self, name):
        r"""Visdom logging function. This function logs the values of the individual gradients.

        Args:
            name (str): Name of the model whose gradients are to be logged.
        """
        self.vis.line(
            [self.logs[name][len(self.logs[name]) - 1]],
            [self.step],
            win=name,
            update="append",
            opts=dict(title=name, xlabel="Time Step", ylabel="Gradient"),
        )

    def update_grads(self, name, model, eps=1e-5):
        r"""Updates the gradient logs.

        Args:
            name (str): Name of the model.
            model (torch.nn.Module): Either a ``torchgan.models.Generator`` or a
                ``torchgan.models.Discriminator`` or their subclass.
            eps (float, optional): Tolerance value.
        """
        gradsum = 0.0
        for p in model.parameters():
            if p.grad is not None:
                gradsum += torch.sum(p.grad ** 2).clone().item()
        if gradsum > eps:
            self.logs[name][len(self.logs[name]) - 1] += gradsum
            model.zero_grad()

    def report_end_epoch(self):
        r"""Prints to the console at the end of the epoch.
        """
        if CONSOLE_LOGGING == 1:
            for key, val in self.logs.items():
                print("{} Mean Gradients : {}".format(key, sum(val) / len(val)))

    def __call__(self, trainer, **kwargs):
        for name in trainer.model_names:
            super(GradientVisualize, self).__call__(name, **kwargs)
            self.logs[name].append(0.0)


class ImageVisualize(Visualize):
    r"""This class provides the Logging for the Images.

    Args:
        trainer (torchgan.trainer.Trainer): The base trainer used for training.
        visdom_port (int, optional): Port to log using ``visdom``. The visdom server needs to be
            manually started at this port else an error will be thrown and the code will crash.
            This is ignored if ``VISDOM_LOGGING`` is ``0``.
        log_dir (str, optional): Directory where TensorboardX should store the logs. This is
            ignored if ``TENSORBOARD_LOGGING`` is ``0``.
        writer (tensorboardX.SummaryWriter, optonal): Send a `SummaryWriter` if you
            don't want to start a new SummaryWriter.
        test_noise (torch.Tensor, optional): If provided then it will be used as the noise for image
            sampling.
        nrow (int, optional): Number of rows in which the image is to be stored.
    """

    def __init__(
        self,
        trainer,
        visdom_port=8097,
        log_dir=None,
        writer=None,
        test_noise=None,
        nrow=8,
    ):
        super(ImageVisualize, self).__init__(
            [], visdom_port=visdom_port, log_dir=log_dir, writer=writer
        )
        self.test_noise = []
        for model in trainer.model_names:
            if isinstance(getattr(trainer, model), Generator):
                self.test_noise.append(
                    getattr(trainer, model).sampler(trainer.sample_size, trainer.device)
                    if test_noise is None
                    else test_noise
                )
        self.step = 1
        self.nrow = nrow

    def log_tensorboard(self, trainer, image, model):
        r"""Logs a generated image in tensorboard at the end of an epoch.

        Args:
            trainer (torchgan.trainer.Trainer): The base trainer used for training.
            image (Image): The generated image.
            model (str): The name of the model which generated the ``image``.
        """
        self.writer.add_image("Generated Samples/{}".format(model), image, self.step)

    def log_console(self, trainer, image, model):
        r"""Saves a generated image at the end of an epoch. The path where the image is
        being stored is controlled by the ``trainer``.

        Args:
            trainer (torchgan.trainer.Trainer): The base trainer used for training.
            image (Image): The generated image.
            model (str): The name of the model which generated the ``image``.
        """
        save_path = "{}/epoch{}_{}.png".format(trainer.recon, self.step, model)
        print("Generating and Saving Images to {}".format(save_path))
        torchvision.utils.save_image(image, save_path)

    def log_visdom(self, trainer, image, model):
        r"""Logs a generated image in visdom at the end of an epoch.

        Args:
            trainer (torchgan.trainer.Trainer): The base trainer used for training.
            image (Image): The generated image.
            model (str): The name of the model which generated the ``image``.
        """
        self.vis.image(image, opts=dict(caption="Generated Samples/{}".format(model)))

    def __call__(self, trainer, **kwargs):
        pos = 0
        for model in trainer.model_names:
            if isinstance(getattr(trainer, model), Generator):
                generator = getattr(trainer, model)
                with torch.no_grad():
                    image = generator(*self.test_noise[pos])
                    image = torchvision.utils.make_grid(
                        image, nrow=self.nrow, normalize=True, range=(-1, 1)
                    )
                    super(ImageVisualize, self).__call__(
                        trainer, image, model, **kwargs
                    )
                self.step -= 1
                pos = pos + 1
        self.step += 1 if pos > 0 else 0
