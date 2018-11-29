import torch
import torchvision
from ..models.model import Generator, Discriminator
from .backends import *
if TENSORBOARD_LOGGING == 1:
    from tensorboardX import SummaryWriter
if VISDOM_LOGGING == 1:
    import visdom

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
    def __init__(self, visualize_list, visdom_port=8097, log_dir=None, writer=None):
        self.logs = {}
        for item in visualize_list:
            name = type(item).__name__
            self.logs[name] = []
        self.step = 1
        if TENSORBOARD_LOGGING == 1:
            self.build_tensorboard(log_dir, writer)
        if VISDOM_LOGGING == 1:
            self.build_visdom(visdom_port)

    def build_tensorboard(self, log_dir, writer):
        self.writer = SummaryWriter(log_dir) if writer is None else writer

    def build_visdom(self, port):
        self.vis = visdom.Visdom(port=port)

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

    def log_visdom(self):
        r"""Visdom logging function. Needs to be defined in the subclass"""
        raise NotImplementedError

    def __call__(self, *args, lock_console=False, lock_tensorboard=False, lock_visdom=False,
                 **kwargs):
        if not lock_console and CONSOLE_LOGGING == 1:
            self.log_console(*args, **kwargs)
        if not lock_tensorboard and TENSORBOARD_LOGGING == 1:
            self.log_tensorboard(*args, **kwargs)
        if not lock_visdom and VISDOM_LOGGING == 1:
            self.log_visdom(*args, **kwargs)
        self.step_update()

class LossVisualize(Visualize):
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

    def log_visdom(self, running_losses):
        self.vis.line([running_losses["Running Discriminator Loss"]], [self.step],
                      win="Running Discriminator Loss", update="append",
                      opts=dict(title="Running Discriminator Loss", xlabel="Time Step",
                      ylabel="Running Loss"))
        self.vis.line([running_losses["Running Generator Loss"]], [self.step],
                      win="Running Generator Loss", update="append",
                      opts=dict(title="Running Generator Loss", xlabel="Time Step",
                      ylabel="Running Loss"))
        self.vis.line([[running_losses["Running Discriminator Loss"],
                      running_losses["Running Generator Loss"]]], [self.step],
                      win="Running Losses", update="append",
                      opts=dict(title="Running Losses", xlabel="Time Step",
                      ylabel="Running Loss", legend=["Discriminator", "Generator"]))
        for name, value in self.logs.items():
            val = value[-1]
            if type(val) is tuple:
                name1 = "{}-Generator".format(name)
                name2 = "{}-Discriminator".format(name)
                self.vis.line([val[0]], [self.step], win=name1, update="append",
                              opts=dict(title=name1, xlabel="Time Step", ylabel="Loss Value"))
                self.vis.line([val[1]], [self.step], win=name2, update="append",
                              opts=dict(title=name2, xlabel="Time Step", ylabel="Loss Value"))
            else:
                self.vis.line([val], [self.step], win=name, update="append",
                              opts=dict(title=name, xlabel="Time Step", ylabel="Loss Value"))

    def __call__(self, trainer, **kwargs):
        running_generator_loss = trainer.loss_information["generator_losses"] /\
            trainer.loss_information["generator_iters"]
        running_discriminator_loss = trainer.loss_information["discriminator_losses"] /\
            trainer.loss_information["discriminator_iters"]
        running_losses = {"Running Discriminator Loss": running_discriminator_loss,
                          "Running Generator Loss": running_generator_loss}
        super(LossVisualize, self).__call__(running_losses, **kwargs)

class MetricVisualize(Visualize):
    def log_tensorboard(self):
        for name, value in self.logs.items():
            self.writer.add_scalar("Metrics/{}".format(name), value[-1], self.step)

    def log_console(self):
        for name, val in self.logs.items():
            print('{} : {}'.format(name, val[-1]))

    def log_visdom(self):
        for name, value in self.logs.items():
            self.vis.line([value[-1]], [self.step], win=name, update="append",
                          opts=dict(title=name, xlabel="Time Step", ylabel="Metric Value"))

class GradientVisualize(Visualize):
    def log_tensorboard(self, name, gradsum):
        self.writer.add_scalar('Gradients/{}'.format(name), gradsum, self.step)

    def log_console(self, name, gradsum):
        print('{} Gradients : {}'.format(name, gradsum))

    def log_visdom(self, name, gradsum):
        self.vis.line([gradsum], [self.step], win=name, update="append",
                      opts=dict(title=name, xlabel="Time Step", ylabel="Gradient"))

    def __call__(self, trainer, **kwargs):
        for name in trainer.model_names:
            model = getattr(trainer, name)
            gradsum = 0.0
            for p in model.parameters():
                gradsum += p.norm(2).item()
            super(GradientVisualize, self).__call__(name, gradsum, **kwargs)

class ImageVisualize(Visualize):
    def __init__(self, trainer, visdom_port=8097, log_dir=None, writer=None, test_noise=None, nrow=8):
        super(ImageVisualize, self).__init__([], visdom_port=visdom_port, log_dir=log_dir, writer=writer)
        self.test_noise = []
        for model in trainer.model_names:
            if isinstance(getattr(trainer, model), Generator):
                self.test_noise.append(getattr(trainer, model).sampler(trainer.sample_size, trainer.device)
                                       if test_noise is None else test_noise)
        self.step = 1
        self.nrow = nrow

    def log_tensorboard(self, trainer, image, model):
        self.writer.add_image("Generated Samples/{}".format(model), image, self.step)

    def log_console(self, trainer, image, model):
        save_path = "{}/epoch{}_{}.png".format(trainer.recon, self.step, model)
        print("Generating and Saving Images to {}".format(save_path))
        torchvision.utils.save_image(image, save_path, nrow=self.nrow)

    def log_visdom(self, trainer, image, model):
        self.vis.image(image, opts=dict(caption="Generated Samples/{}".format(model)))

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
