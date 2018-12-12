from .visualize import *
from .backends import *
if TENSORBOARD_LOGGING == 1:
    from tensorboardX import SummaryWriter
if VISDOM_LOGGING == 1:
    import visdom

__all__ = ['Logger']

class Logger(object):
    def __init__(self, trainer, losses_list, metrics_list=None, visdom_port=8097,
                 log_dir=None, writer=None, nrow=8, test_noise=None):
        if TENSORBOARD_LOGGING == 1:
            self.writer = SummaryWriter(log_dir) if writer is None else writer
        else:
            self.writer = None
        self.logger_end_epoch = []
        self.logger_mid_epoch = []
        self.logger_end_epoch.append(ImageVisualize(trainer, writer=self.writer, test_noise=test_noise,
                                                    nrow=nrow))
        self.logger_mid_epoch.append(GradientVisualize(trainer.model_names, writer=self.writer))
        if metrics_list is not None:
            self.logger_end_epoch.append(MetricVisualize(metrics_list, writer=self.writer))
        self.logger_mid_epoch.append(LossVisualize(losses_list, writer=self.writer))

    def get_loss_viz(self):
        return self.logger_mid_epoch[1]

    def get_metric_viz(self):
        return self.logger_end_epoch[0]

    def get_grad_viz(self):
        return self.logger_mid_epoch[0]

    def register(self, visualize, *args, mid_epoch=True, **kwargs):
        if mid_epoch:
            self.logger_mid_epoch.append(visualize(*args, writer=self.writer, **kwargs))
        else:
            self.logger_end_epoch.append(visualize(*args, writer=self.writer, **kwargs))

    def close(self):
        self.writer.close()

    def run_mid_epoch(self, trainer, *args):
        for logger in self.logger_mid_epoch:
            if type(logger).__name__ is "LossVisualize" or\
               type(logger).__name__ is "GradientVisualize":
                logger(trainer, lock_console=True)
            else:
                logger(*args, lock_console=True)

    def run_end_epoch(self, trainer, epoch, *args):
        print("Epoch {} Summary".format(epoch))
        for logger in self.logger_mid_epoch:
            if type(logger).__name__ is "LossVisualize":
                logger(trainer)
            elif type(logger).__name__ is "GradientVisualize":
                logger.report_end_epoch()
            else:
                logger(*args)
        for logger in self.logger_end_epoch:
            if type(logger).__name__ is "ImageVisualize":
                logger(trainer)
            elif type(logger).__name__ is "MetricVisualize":
                logger()
            else:
                logger(*args)
