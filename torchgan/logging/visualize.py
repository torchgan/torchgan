import torch
import torchvision
from ..models.model import Generator, Discriminator
from .backends import *
if TENSORBOARD_LOGGING == 1:
    from tensorboardX import SummaryWriter
if MATPLOTLIB_LOGGING == 1:
    import matplotlib.pyplot as plt
if SEABORN_LOGGING == 1:
    import seaborn as sns

__all__ = ['Visualize', 'LossVisualize', 'MetricVisualize',
           'GradientVisualize', 'ImageVisualize']

class Visualize(object):
    r"""Base class for all Visualizations. Supports printing to Tensorboard and
    Console.

    Args:
        visualize_list (list, optional): List of the functions needed for visualization
        tensorboard (bool, optional): Set it to `True` for using TensorboardX.
        log_dir (str, optional): Directory where TensorboardX should store the logs.
        writer (tensorboardX.SummaryWriter, optonal): Send a `SummaryWriter` if you
                                                      don't want to start a new SummaryWriter.
    """
    def __init__(self, visualize_list, plot_title, plot_xlabel, plot_ylabel, log_dir=None, writer=None):
        self.logs = {}
        for item in visualize_list:
            name = type(item).__name__
            self.logs[name] = []
        self.step = 1
        if TENSORBOARD_LOGGING == 1:
            self.build_tensorboard(log_dir, writer)
        if MATPLOTLIB_LOGGING == 1:
            self.build_matplotlib(plot_title, plot_xlabel, plot_ylabel)
        if SEABORN_LOGGING == 1:
            self.build_seaborn(plot_title, plot_xlabel, plot_ylabel)

    def build_tensorboard(self, log_dir, writer):
        self.writer = SummaryWriter(log_dir) if writer is None else writer

    def build_matplotlib(self, plot_title, plot_xlabel, plot_ylabel):
        self.plots = {}
        self.lines = {}
        for i, title in enumerate(plot_title):
            plot = plt.figure()
            self.plots[title] = plot.gca()
            self.plots[title].set_title(title)
            self.plots[title].set_xlabel(plot_xlabel[i])
            self.plots[title].set_ylabel(plot_ylabel[i])
            self.lines[title] = self.plots[title].plot([], [])[0]
        for title in self.logs.keys():
            plot = plt.figure()
            self.plots[title] = plot.gca()
            self.plots[title].set_title(title)
            self.plots[title].set_xlabel("Time Step")
            self.plots[title].set_ylabel(title)
            self.lines[title] = self.plots[title].plot([], [])[0]

    def build_seaborn(self):
        pass

    def step_update(self):
        r"""Helper function which updates the step at the end of
        one print iteration.
        """
        self.step += 1

    def log_tensorboard(self):
        r"""Tensorboard logging function. Needs to be defined in the subclass"""
        raise NotImplementedError

    def log_console(self):
        r"""Console logging function. Needs to be defined in the subclass"""
        raise NotImplementedError

    def log_matplotlib(self):
        r"""Matplotlib logging function. Needs to be defined in the subclass"""
        raise NotImplementedError

    def log_seaborn(self):
        r"""Seaborn logging function. Needs to be defined in the subclass"""
        raise NotImplementedError

    def __call__(self, *args, lock_console=False, lock_tensorboard=False, lock_matplotlib=False,
                 lock_seaborn=False, **kwargs):
        if not lock_console and CONSOLE_LOGGING == 1:
            self.log_console(*args, **kwargs)
        if not lock_tensorboard and TENSORBOARD_LOGGING == 1:
            self.log_tensorboard(*args, **kwargs)
        if not lock_matplotlib and MATPLOTLIB_LOGGING == 1:
            self.log_matplotlib(*args, **kwargs)
        if not lock_seaborn and SEABORN_LOGGING == 1:
            self.log_seaborn(*args, **kwargs)
        self.step_update()

class LossVisualize(Visualize):
    def __init__(self, *args, **kwargs):
        super(LossVisualize, self).__init__(*args, ["Running Discriminator Loss", "Running Generator Loss"],
                                            ["Time Step", "Time Step"], ["Loss Value", "Loss Value"], **kwargs)

    def log_tensorboard(self, running_losses):
        self.writer.add_scalar("Running Discriminator Loss",
                               running_losses["Running Discriminator Loss"],
                               self.step)
        self.writer.add_scalar("Running Generator Loss",
                               running_losses["Running Generator Loss"],
                               self.step)
        self.writer.add_scalars("Running Losses",
                                running_losses,
                                self.step)
        for name, value in self.logs.items():
            val = value[-1]
            if type(val) is tuple:
                self.writer.add_scalar('Losses/{}-Generator'.format(name), val[0], self.step)
                self.writer.add_scalar('Losses/{}-Discriminator'.format(name), val[1], self.step)
            else:
                self.writer.add_scalar('Losses/{}'.format(name), val, self.step)

    def log_console(self, running_losses):
        for name, val in running_losses.items():
            print('Mean {} : {}'.format(name, val))

    def log_matplotlib(self, running_losses):
        self.lines["Running Discriminator Loss"].set_data(self.step,
                                                          running_losses["Running Discriminator Loss"])
        self.lines["Running Generator Loss"].set_data(self.step,
                                                      running_losses["Running Generator Loss"])
        # NOTE(avik-pal): Need to find out a way to plot 2 lines in 1 graph
        for name, value in self.logs.items():
            val = value[-1]
            # NOTE(avik-pal): Tuple case not handled
            self.lines[name].set_data(self.step, val)
        for name, plot in self.plots.items():
            plot.show(block=False)

    def log_seaborn(self, running_losses):
        pass

    def __call__(self, trainer, **kwargs):
        running_generator_loss = trainer.loss_information["generator_losses"] /\
            trainer.loss_information["generator_iters"]
        running_discriminator_loss = trainer.loss_information["discriminator_losses"] /\
            trainer.loss_information["discriminator_iters"]
        running_losses = {"Running Discriminator Loss": running_discriminator_loss,
                          "Running Generator Loss": running_generator_loss}
        super(LossVisualize, self).__call__(running_losses, **kwargs)

class MetricVisualize(Visualize):
    def __init__(self, *args, **kwargs):
        super(MetricVisualize, self).__init__(*args, [], [], [], **kwargs)

    def log_tensorboard(self):
        for name, value in self.logs.items():
            self.writer.add_scalar("Metrics/{}".format(name), value[-1], self.step)

    def log_console(self):
        for name, val in self.logs.items():
            print('{} : {}'.format(name, val[-1]))

    def log_matplotlib(self):
        for name, value in self.logs.items():
            self.lines[name].set_data(self.step, value[-1])
        for name, plot in self.plots.items():
            plot.show(block=False)

    def log_seaborn(self):
        pass

class GradientVisualize(Visualize):
    def __init__(self, *args, model_names=None, **kwargs):
        super(GradientVisualize, self).__init__(*args, [], [], [], **kwargs)
        if MATPLOTLIB_LOGGING == 1:
            self.build_matplotlib_custom(model_names)

    def build_matplotlib_custom(self, model_names):
        self.plots = {}
        self.lines = {}
        for name in model_names:
            plot = plt.figure()
            self.plots[name] = plot.gca()
            self.plots[name].set_title(name)
            self.plots[name].set_xlabel("Time Step")
            self.plots[name].set_ylabel("Gradient")
            self.lines[name] = self.plots[name].plot([], [])[0]

    def log_tensorboard(self, name, gradsum):
        self.writer.add_scalar('Gradients/{}'.format(name), gradsum, self.step)

    def log_console(self, name, gradsum):
        print('{} Gradients : {}'.format(name, gradsum))

    def log_matplotlib(self, name, gradsum):
        self.lines[name].set_data(self.step, gradsum)
        plt.show(self.plots[name], block=False)

    def log_seaborn(self, name, gradsum):
        pass

    def __call__(self, trainer, **kwargs):
        for name in trainer.model_names:
            model = getattr(trainer, name)
            gradsum = 0.0
            for p in model.parameters():
                gradsum += p.norm(2).item()
            super(GradientVisualize, self).__call__(name, gradsum, **kwargs)

class ImageVisualize(Visualize):
    def __init__(self, trainer, log_dir=None, writer=None, test_noise=None, nrow=8):
        self.test_noise = []
        for model in trainer.model_names:
            if isinstance(getattr(trainer, model), Generator):
                self.test_noise.append(getattr(trainer, model).sampler(trainer.sample_size, trainer.device)
                                       if test_noise is None else test_noise)
        self.step = 1
        self.nrow = nrow
        if TENSORBOARD_LOGGING == 1:
            self.build_tensorboard(log_dir, writer)

    def log_tensorboard(self, trainer, image, model):
        self.writer.add_image("Generated Samples/{}".format(model), image, self.step)

    def log_console(self, trainer, image, model):
        save_path = "{}/epoch{}_{}.png".format(trainer.recon, self.step, model)
        print("Generating and Saving Images to {}".format(save_path))
        torchvision.utils.save_image(image, save_path, nrow=self.nrow)

    def log_matplotlib(self, trainer, image, model):
        pass

    def log_seaborn(self, trainer, image, model):
        pass

    def __call__(self, trainer, **kwargs):
        pos = 0
        for model in trainer.model_names:
            if isinstance(getattr(trainer, model), Generator):
                generator = getattr(trainer, model)
                with torch.no_grad():
                    image = generator(*self.test_noise[pos])
                    image = torchvision.utils.make_grid(image)
                    super(ImageVisualize, self).__call__(trainer, image, model, **kwargs)
                pos = pos + 1
