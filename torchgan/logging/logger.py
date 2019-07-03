from .backends import *
from .visualize import *

if TENSORBOARD_LOGGING == 1:
    from tensorboardX import SummaryWriter
if VISDOM_LOGGING == 1:
    import visdom

__all__ = ["Logger"]


class Logger(object):
    r"""Base Logger class. It controls the executions of all the Visualizers and is deeply
    integrated with the functioning of the Trainer.

    .. note::
        The ``Logger`` has been designed to be controlled internally by the ``Trainer``. It is
        recommended that the user does not attempt to use it externally in any form.

    .. warning::
        This ``Logger`` is meant to work on the standard Visualizers available. Work is being
        done to support custom Visualizers in a clean way. But currently it is not possible to
        do so.

    Args:
        trainer (torchgan.trainer.Trainer): The base trainer used for training.
        losses_list (list): A list of the Loss Functions that need to be minimized. For a list of
            pre-defined losses look at :mod:`torchgan.losses`. All losses in the list must be a
            subclass of atleast ``GeneratorLoss`` or ``DiscriminatorLoss``.
        metrics_list (list, optional): List of Metric Functions that need to be logged. For a list of
            pre-defined metrics look at :mod:`torchgan.metrics`. All losses in the list must be a
            subclass of ``EvaluationMetric``.
        visdom_port (int, optional): Port to log using ``visdom``. A deafult server is started
            at port ``8097``. So manually a new server has to be started if the post is changed.
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
        losses_list,
        metrics_list=None,
        visdom_port=8097,
        log_dir=None,
        writer=None,
        nrow=8,
        test_noise=None,
    ):
        if TENSORBOARD_LOGGING == 1:
            self.writer = SummaryWriter(log_dir) if writer is None else writer
        else:
            self.writer = None
        self.logger_end_epoch = []
        self.logger_mid_epoch = []
        self.logger_end_epoch.append(
            ImageVisualize(
                trainer, writer=self.writer, test_noise=test_noise, nrow=nrow
            )
        )
        self.logger_mid_epoch.append(
            GradientVisualize(trainer.model_names, writer=self.writer)
        )
        if metrics_list is not None:
            self.logger_end_epoch.append(
                MetricVisualize(metrics_list, writer=self.writer)
            )
        self.logger_mid_epoch.append(LossVisualize(losses_list, writer=self.writer))

    def get_loss_viz(self):
        r"""Get the LossVisualize object.
        """
        return self.logger_mid_epoch[1]

    def get_metric_viz(self):
        r"""Get the MetricVisualize object.
        """
        return self.logger_end_epoch[1]

    def get_grad_viz(self):
        r"""Get the GradientVisualize object.
        """
        return self.logger_mid_epoch[0]

    def register(self, visualize, *args, mid_epoch=True, **kwargs):
        r"""Register a new ``Visualize`` object with the Logger.

        Args:
            visualize (torchgan.logging.Visualize): Class name of the visualizer.
            mid_epoch (bool, optional): Set it to ``False`` if it is to be executed once the epoch is
                over. Otherwise it is executed after every call to the ``train_iter``.
        """
        if mid_epoch:
            self.logger_mid_epoch.append(visualize(*args, writer=self.writer, **kwargs))
        else:
            self.logger_end_epoch.append(visualize(*args, writer=self.writer, **kwargs))

    def close(self):
        r"""Turns off the tensorboard ``SummaryWriter`` if it were created.
        """
        if self.writer is not None:
            self.writer.close()

    def run_mid_epoch(self, trainer, *args):
        r"""Runs the Visualizers after every call to the ``train_iter``.

        Args:
            trainer (torchgan.trainer.Trainer): The base trainer used for training.
        """
        for logger in self.logger_mid_epoch:
            if (
                type(logger).__name__ == "LossVisualize"
                or type(logger).__name__ == "GradientVisualize"
            ):
                logger(trainer, lock_console=True)
            else:
                logger(*args, lock_console=True)

    def run_end_epoch(self, trainer, epoch, time_duration, *args):
        r"""Runs the Visualizers at the end of one epoch.

        Args:
            trainer (torchgan.trainer.Trainer): The base trainer used for training.
            epoch (int): The epoch number which was completed.
        """
        print("Epoch {} Summary".format(epoch + 1))
        print("Epoch time duration : {}".format(time_duration))
        for logger in self.logger_mid_epoch:
            if type(logger).__name__ == "LossVisualize":
                logger(trainer)
            elif type(logger).__name__ == "GradientVisualize":
                logger.report_end_epoch()
            else:
                logger(*args)
        for logger in self.logger_end_epoch:
            if type(logger).__name__ == "ImageVisualize":
                logger(trainer)
            elif type(logger).__name__ == "MetricVisualize":
                logger()
            else:
                logger(*args)
        print()
