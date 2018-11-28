from .visualize import *
from tensorboardX import SummaryWriter

__all__ = ['Logger']

class Logger(object):
    def __init__(self, trainer, losses_list, metrics_list=None, tensorboard=True,
                 log_dir=None, writer=None, nrow=8, test_noise=None, no_print=False):
        self.writer = SummaryWriter(log_dir) if writer is None else writer
        self.logger_end_epoch = []
        self.logger_mid_epoch = []
        self.logger_end_epoch.append(ImageVisualize(trainer, tensorboard=tensorboard,
                                                    writer=self.writer, test_noise=test_noise,
                                                    nrow=nrow))
        self.logger_mid_epoch.append(GradientVisualize([], tensorboard=tensorboard, writer=self.writer))
        if metrics_list is not None:
            self.logger_end_epoch.append(MetricVisualize(metrics_list, tensorboard=tensorboard,
                                                         writer=self.writer))
        self.logger_mid_epoch.append(LossVisualize(losses_list, tensorboard=tensorboard, writer=self.writer))
        self.no_print = no_print

    def get_loss_viz(self):
        return self.logger_mid_epoch[1]

    def get_metric_viz(self):
        return self.logger_end_epoch[0]

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
                logger(trainer, lock_items=True)
            else:
                logger(*args, lock_items=True)

    def run_end_epoch(self, trainer, epoch, *args):
        if not self.no_print:
            print("Epoch {} Summary".format(epoch))
        for logger in self.logger_mid_epoch:
            if type(logger).__name__ is "LossVisualize" or\
               type(logger).__name__ is "GradientVisualize":
                logger(trainer, lock_items=self.no_print)
            else:
                logger(*args, lock_items=self.no_print)
        for logger in self.logger_end_epoch:
            if type(logger).__name__ is "ImageVisualize":
                logger(trainer, lock_items=self.no_print)
            elif type(logger).__name__ is "MetricVisualize":
                logger(lock_items=self.no_print)
            else:
                logger(*args, lock_items=self.no_print)
