import os
import time
from types import *
import logging as lg
import warnings
from inspect import _empty, signature
from warnings import warn

import torch
import torchvision

from torchgan.losses import DiscriminatorLoss, GeneratorLoss
from torchgan.models import Discriminator, Generator

import pytorch_lightning as pl


class LightningGANModule(pl.LightningModule):
    r"""Trainer for TorchGAN built on top of Pytorch Lightning. This shall be the
    default trainer post 0.1 release and all other trainers shall be deprecated.
    """

    def __init__(
        self,
        models_config,
        losses_list,
        train_dataloader,
        val_dataloader=None,
        metrics_list=None,
        ncritic=1,
        sample_size=8,
        test_noise=None,
        **kwargs,
    ):
        super().__init__()

        self.model_names = []
        self.optimizer_names = []
        self.schedulers = []

        for key, model_config in models_config.items():
            # Model creation and storage
            self.model_names.append(key)
            if "model" in model_config:
                setattr(self, key, model_config["model"])
            elif "args" in model_config or "name" in model_config:
                warnings.warn(
                    "This is the old TorchGAN API. It is deprecated"
                    + "and shall be removed post v0.1. Please update to"
                    + "instantiating the model and pass it using the key"
                    + "`model`",
                    FutureWarning,
                )
                args = model_config.get("args", {})
                # Instantiate a GAN model
                setattr(self, key, (model_config["name"])(**args))
            else:
                raise Exception(
                    f"Couldn't find/instantiate the model corresponding to"
                    + f"{key}"
                )
            model = getattr(self, key)

            # Dealing with the optimizers
            opt = model_config.get("optimizer", {})
            if type(opt) is dict:
                if "optimizer" not in model_config:
                    warnings.warn(
                        "This is the old TorchGAN API. It is deprecated"
                        + "and shall be removed post v0.1. Please update to"
                        + "creating a lambda function taking as input"
                        + "the model parameters",
                        FutureWarning,
                    )
                opt_name = opt.get("var", f"optimizer_{key}")
                self.optimizer_names.append(opt_name)
                setattr(
                    self,
                    opt_name,
                    opt.get("name", torch.optim.Adam)(
                        model.parameters(), **opt.get("args", {})
                    ),
                )
            elif type(opt) is FunctionType:
                opt_name = opt.get("optimizer_name", f"optimizer_{key}")
                self.optimizer_names.append(opt_name)
                setattr(self, opt_name, opt(model.parameters()))
            else:
                raise Exception(
                    f"Couldn't find/instantiate the optimizer corresponding to"
                    + f"{key}"
                )

            # TODO: Deal with schedulers

        self.losses = {}
        for loss in losses_list:
            self.losses[type(loss).__name__] = loss

        if metrics_list is None:
            self.metrics = None
        else:
            self.metrics = {}
            for metric in metrics_list:
                self.metrics[type(metric).__name__] = metric

        self.sample_size = sample_size

        # Not needed but we need to store this to avoid errors.
        # Also makes life simpler
        self.noise = None
        self.real_inputs = None
        self.labels = None

        self.generator_steps = 0
        self.discriminator_steps = 0

        assert ncritic != 0
        if ncritic > 0:
            self.ncritic = ncritic
            self.ngen = 1
        else:
            self.ncritic = 1
            self.ngen = abs(ncritic)

        # This exists for convenience. We will handle the device from data in
        # the `training_step` function
        self.device = torch.device("cpu")

        for key, val in kwargs.items():
            if key in self.__dict__:
                warn(
                    "Overiding the default value of {} from {} to {}".format(
                        key, getattr(self, key), val
                    )
                )
            setattr(self, key, val)

        # This is only temporarily stored and is deleted
        self.train_dataloader_cached = train_dataloader
        self.val_dataloader_cached = val_dataloader

        self._store_loss_maps()
        self._store_metric_maps()


    def configure_optimizers(self):
        optimizers = [getattr(self, name) for name in self.optimizer_names]
        return optimizers  # , self.schedulers

    @pl.data_loader
    def train_dataloader(self):
        train_dataloader_cached = self.train_dataloader_cached
        return train_dataloader_cached

    def _get_argument_maps(self, default_map, func):
        r"""Extracts the signature of the `func`. Then it returns the list of
        arguments that are present in the object and need to be mapped and
        passed to the `func` when calling it.

        Args:
            default_map (dict): The keys of this dictionary override the
                                function signature.
            func (function): Function whose argument map is to be generated.

        Returns:
            List of arguments that need to be fed into the function. It contains
            all the positional arguments and keyword arguments that are stored
            in the object. If any of the required arguments are not present an
            error is thrown.
        """
        sig = signature(func)
        arg_map = {}
        for sig_param in sig.parameters.values():
            arg = sig_param.name
            arg_name = arg
            if arg in default_map:
                arg_name = default_map[arg]
            if sig_param.default is not _empty:
                if arg_name in self.__dict__ or arg_name in self.__dict__["_modules"]:
                    arg_map.update({arg: arg_name})
            else:
                if (
                    arg_name not in self.__dict__
                    and arg_name not in self.__dict__["_modules"]
                    and arg != "kwargs"
                    and arg != "args"
                ):
                    raise Exception(
                        "Argument : {} not present.".format(arg_name)
                    )
                else:
                    arg_map.update({arg: arg_name})
        return arg_map

    def _store_metric_maps(self):
        r"""Creates a mapping between the metrics and the arguments from the object that need to be
        passed to it.
        """
        if self.metrics is not None:
            self.metric_arg_maps = {}
            for name, metric in self.metrics.items():
                self.metric_arg_maps[name] = self._get_argument_maps(
                    metric.arg_map, metric.metric_ops
                )

    def _store_loss_maps(self):
        r"""Creates a mapping between the losses and the arguments from the object that need to be
        passed to it.
        """
        self.loss_arg_maps = {}
        for name, loss in self.losses.items():
            self.loss_arg_maps[name] = self._get_argument_maps(
                loss.arg_map, loss.train_ops
            )

    def _get_arguments(self, arg_map):
        r"""Get the argument values from the object and create a dictionary.

        Args:
            arg_map (dict): A dict of arguments that is generated by `_get_argument_maps`.

        Returns:
            A dictionary mapping the argument name to the value of the argument.
        """
        args = {}
        for key, val in arg_map.items():
            if val == "device":
                args[key] = self._get_device_from_tensor(self.real_inputs)
                continue
            if val in self.__dict__:
                args[key] = self.__dict__[val]
            else:
                args[key] = self.__dict__["_modules"][val]
        return args

    def optimizer_step(
        self,
        current_epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        second_order_closure=None,
    ):
        # We handle the optimizer step and zero_grad in the train_ops
        # itself, so override Pytorch Lightning's default function
        return

    def handle_data_batch(self, batch):
        if type(batch) in (tuple, list):
            self.real_inputs = batch[0]
            self.labels = batch[1]
        elif type(batch) is torch.Tensor:
            self.real_inputs = batch
        else:
            self.real_inputs = batch

    def _unfreeze_parameters(self):
        for name in self.model_names:
            model = getattr(self, name)
            for param in model.parameters():
                param.requires_grad = True

    def _get_device_from_tensor(self, x: torch.Tensor):
        if self.on_gpu:
            device = torch.device(f"cuda:{x.device.index}")
            return device
        return torch.device("cpu")

    def training_step(self, batch, batch_idx, optimizer_idx):
        self.handle_data_batch(batch)
        # FIXME: PyLightning seems to convert all the parameters to
        #        require no grad.
        self._unfreeze_parameters()

        gen_loss = 0.0
        dis_loss = 0.0

        train_gen = self.discriminator_steps % self.ncritic == 0
        train_dis = self.generator_steps % self.ngen == 0

        for name, loss in self.losses.items():
            lgen = isinstance(loss, GeneratorLoss)
            ldis = isinstance(loss, DiscriminatorLoss)

            if lgen and ldis:
                if train_dis:
                    cur_loss = loss.train_ops(
                        **self._get_arguments(self.loss_arg_maps[name])
                    )

                    if type(cur_loss) in (tuple, list):
                        gen_loss += cur_loss[0]
                        self.generator_steps += 1
                        dis_loss += cur_loss[1]
                        self.discriminator_steps += 1
                    else:
                        dis_loss += cur_loss
                        self.discriminator_steps += 1
            elif lgen:
                if train_gen:
                    cur_loss = loss.train_ops(
                        **self._get_arguments(self.loss_arg_maps[name])
                    )

                    gen_loss += cur_loss
                    self.generator_steps += 1
            elif ldis:
                if train_dis:
                    cur_loss = loss.train_ops(
                        **self._get_arguments(self.loss_arg_maps[name])
                    )

                    dis_loss += cur_loss
                    self.discriminator_steps += 1
            else:
                raise Exception(
                    f"type({loss}) is {type(loss)} which is not a subclass"
                    + f"of GeneratorLoss / DiscriminatorLoss"
                )
        # Bypass Lightning by passing a zero loss
        loss = torch.zeros(1)
        loss.requires_grad = True
        return {
            "loss": loss,
            "progress_bar": {"Generator Loss": gen_loss, "DiscriminatorLoss": dis_loss},
        }

    def forward(x):
        pass

