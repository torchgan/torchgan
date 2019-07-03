import torch

from ..logging.logger import Logger
from ..losses.loss import DiscriminatorLoss, GeneratorLoss
from ..models.model import Discriminator, Generator
from .base_trainer import BaseTrainer

__all__ = ["ParallelTrainer"]


class ParallelTrainer(BaseTrainer):
    r"""MultiGPU Trainer for GANs. Use the ``Trainer`` class for training on a single GPU or a CPU
    machine.

    Args:
        models (dict): A dictionary containing a mapping between the variable name, storing the
            ``generator``, ``discriminator`` and any other model that you might want to define, with the
            function and arguments that are needed to construct the model. Refer to the examples to
            see how to define complex models using this API.
        losses_list (list): A list of the Loss Functions that need to be minimized. For a list of
            pre-defined losses look at :mod:`torchgan.losses`. All losses in the list must be a
            subclass of atleast ``GeneratorLoss`` or ``DiscriminatorLoss``.
        devices (list): Devices in which the operations are to be carried out. If you
            are using a CPU machine or a single GPU machine use the Trainer class.
        metrics_list (list, optional): List of Metric Functions that need to be logged. For a list of
            pre-defined metrics look at :mod:`torchgan.metrics`. All losses in the list must be a
            subclass of ``EvaluationMetric``.
        ncritic (int, optional): Setting it to a value will make the discriminator train that many
            times more than the generator. If it is set to a negative value the generator will be
            trained that many times more than the discriminator.
        sample_size (int, optional): Total number of images to be generated at the end of an epoch
            for logging purposes.
        epochs (int, optional): Total number of epochs for which the models are to be trained.
        checkpoints (str, optional): Path where the models are to be saved. The naming convention is
            if checkpoints is ``./model/gan`` then models are saved as ``./model/gan0.model`` and so on.
        retain_checkpoints (int, optional): Total number of checkpoints that should be retained. For
            example, if the value is set to 3, we save at most 3 models and start rewriting the models
            after that.
        recon (str, optional): Directory where the sampled images are saved. Make sure the directory
            exists from beforehand.
        log_dir (str, optional): The directory for logging tensorboard. It is ignored if
            TENSORBOARD_LOGGING is 0.
        test_noise (torch.Tensor, optional): If provided then it will be used as the noise for image
            sampling.
        nrow (int, optional): Number of rows in which the image is to be stored.

    Any other argument that you need to store in the object can be simply passed via keyword arguments.

    Example:
        >>> dcgan = ParallelTrainer(
                    {"generator": {"name": DCGANGenerator, "args": {"out_channels": 1, "step_channels":
                                   16}, "optimizer": {"name": Adam, "args": {"lr": 0.0002,
                                   "betas": (0.5, 0.999)}}},
                     "discriminator": {"name": DCGANDiscriminator, "args": {"in_channels": 1,
                                       "step_channels": 16}, "optimizer": {"var": "opt_discriminator",
                                       "name": Adam, "args": {"lr": 0.0002, "betas": (0.5, 0.999)}}}},
                    [MinimaxGeneratorLoss(), MinimaxDiscriminatorLoss()],
                    [0, 1, 2],
                    sample_size=64, epochs=20)
    """

    def __init__(
        self,
        models,
        losses_list,
        devices,
        metrics_list=None,
        ncritic=1,
        epochs=5,
        sample_size=8,
        checkpoints="./model/gan",
        retain_checkpoints=5,
        recon="./images",
        log_dir=None,
        test_noise=None,
        nrow=8,
        **kwargs
    ):
        super(ParallelTrainer, self).__init__(
            losses_list,
            metrics_list=metrics_list,
            device=devices[0],
            ncritic=ncritic,
            epochs=epochs,
            sample_size=sample_size,
            checkpoints=checkpoints,
            retain_checkpoints=retain_checkpoints,
            recon=recon,
            log_dir=log_dir,
            test_noise=test_noise,
            nrow=nrow,
            **kwargs
        )

        self.devices = devices
        self.model_names = []
        self.optimizer_names = []
        self.schedulers = []
        for key, model in models.items():
            self.model_names.append(key)
            if "args" in model:
                setattr(self, key, (model["name"](**model["args"])).to(self.device))
            else:
                setattr(self, key, (model["name"]()).to(self.device))
            for m in getattr(self, key)._modules:
                getattr(self, key)._modules[m] = torch.nn.DataParallel(
                    getattr(self, key)._modules[m], device_ids=devices
                )
            opt = model["optimizer"]
            opt_name = "optimizer_{}".format(key)
            if "var" in opt:
                opt_name = opt["var"]
            self.optimizer_names.append(opt_name)
            model_params = getattr(self, key).parameters()
            if "args" in opt:
                setattr(self, opt_name, (opt["name"](model_params, **opt["args"])))
            else:
                setattr(self, opt_name, (opt["name"](model_params)))
            if "scheduler" in opt:
                sched = opt["scheduler"]
                if "args" in sched:
                    self.schedulers.append(
                        sched["name"](getattr(self, opt_name), **sched["args"])
                    )
                else:
                    self.schedulers.append(sched["name"](getattr(self, opt_name)))

        self.logger = Logger(
            self,
            losses_list,
            metrics_list,
            log_dir=log_dir,
            nrow=nrow,
            test_noise=test_noise,
        )

        self._store_loss_maps()
        self._store_metric_maps()
