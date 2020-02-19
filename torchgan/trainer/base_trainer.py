import os
import time
from inspect import _empty, signature
from warnings import warn

import torch
import torchvision

from ..logging.logger import Logger
from ..losses.loss import DiscriminatorLoss, GeneratorLoss
from ..models.model import Discriminator, Generator

__all__ = ["BaseTrainer"]


class BaseTrainer(object):
    r"""Base Trainer for TorchGANs.

    .. warning::
        This trainer is meant to form the base for all other Trainers. This is not meant for direct usage.

    Features provided by this Base Trainer are:

    - Loss and Metrics Logging via the ``Logger`` class.
    - Generating Image Samples.
    - Saving models at the end of every epoch and loading of previously saved models.
    - Highly flexible and allows changing hyperparameters by simply adjusting the arguments.

    Most of the functionalities provided by the Trainer are flexible enough and can be customized by
    simply passing different arguments. You can train anything from a simple DCGAN to complex CycleGANs
    without ever having to subclass this ``Trainer``.

    Args:
        losses_list (list): A list of the Loss Functions that need to be minimized. For a list of
            pre-defined losses look at :mod:`torchgan.losses`. All losses in the list must be a
            subclass of atleast ``GeneratorLoss`` or ``DiscriminatorLoss``.
        metrics_list (list, optional): List of Metric Functions that need to be logged. For a list of
            pre-defined metrics look at :mod:`torchgan.metrics`. All losses in the list must be a
            subclass of ``EvaluationMetric``.
        device (torch.device, optional): Device in which the operation is to be carried out. If you
            are using a CPU machine make sure that you change it for proper functioning.
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
    """

    def __init__(
        self,
        losses_list,
        metrics_list=None,
        device=torch.device("cuda:0"),
        ncritic=1,
        epochs=5,
        sample_size=8,
        checkpoints="./model/gan",
        retain_checkpoints=5,
        recon="./images",
        log_dir=None,
        test_noise=None,
        nrow=8,
        **kwargs,
    ):
        self.device = device
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
        self.epochs = epochs
        self.checkpoints = checkpoints
        self.retain_checkpoints = retain_checkpoints
        self.recon = recon

        # Not needed but we need to store this to avoid errors. Also makes life simpler
        self.noise = None
        self.real_inputs = None
        self.labels = None
        self.batch_size = 1

        self.loss_information = {
            "generator_losses": 0.0,
            "discriminator_losses": 0.0,
            "generator_iters": 0,
            "discriminator_iters": 0,
        }

        assert ncritic != 0
        if ncritic > 0:
            self.ncritic = ncritic
            self.ngen = 1
        else:
            self.ncritic = 1
            self.ngen = abs(ncritic)

        self.start_epoch = 0
        self.last_retained_checkpoint = 0
        for key, val in kwargs.items():
            if key in self.__dict__:
                warn(
                    "Overiding the default value of {} from {} to {}".format(
                        key, getattr(self, key), val
                    )
                )
            setattr(self, key, val)

        os.makedirs(self.checkpoints.rsplit("/", 1)[0], exist_ok=True)
        os.makedirs(self.recon, exist_ok=True)

    def save_model(self, epoch, save_items=None):
        r"""Function saves the model and some necessary information along with it. List of items
        stored for future reference:

        - Epoch
        - Model States
        - Optimizer States
        - Loss Information
        - Loss Objects
        - Metric Objects
        - Loss Logs

        The save location is printed when this function is called.

        Args:
            epoch (int, optional): Epoch Number at which the model is being saved
            save_items (str, list, optional): Pass the variable name of any other item you want to save.
                The item must be present in the `__dict__` else training will come to an abrupt end.
        """
        if self.last_retained_checkpoint == self.retain_checkpoints:
            self.last_retained_checkpoint = 0
        save_path = self.checkpoints + str(self.last_retained_checkpoint) + ".model"
        self.last_retained_checkpoint += 1
        print("Saving Model at '{}'".format(save_path))
        model = {
            "epoch": epoch + 1,
            "loss_information": self.loss_information,
            "loss_objects": self.losses,
            "metric_objects": self.metrics,
            "loss_logs": (self.logger.get_loss_viz()).logs,
        }
        for save_item in self.model_names + self.optimizer_names:
            model.update({save_item: (getattr(self, save_item)).state_dict()})
        if save_items is not None:
            if type(save_items) is list:
                for itms in save_items:
                    model.update({itms: getattr(self, itms)})
            else:
                model.update({save_items: getattr(self, save_items)})
        torch.save(model, save_path)

    def load_model(self, load_path="", load_items=None):
        r"""Function to load the model and some necessary information along with it. List of items
        loaded:

        - Epoch
        - Model States
        - Optimizer States
        - Loss Information
        - Loss Objects
        - Metric Objects
        - Loss Logs

        .. warning::
            An Exception is raised if the model could not be loaded. Make sure that the model being
            loaded was saved previously by ``torchgan Trainer`` itself. We currently do not support
            loading any other form of models but this might be improved in the future.

        Args:
            load_path (str, optional): Path from which the model is to be loaded.
            load_items (str, list, optional): Pass the variable name of any other item you want to load.
                If the item cannot be found then a warning will be thrown and model will start to train
                from scratch. So make sure that item was saved.
        """
        if load_path == "":
            load_path = self.checkpoints + str(self.last_retained_checkpoint) + ".model"
        print("Loading Model From '{}'".format(load_path))
        try:
            checkpoint = torch.load(load_path)
            self.start_epoch = checkpoint["epoch"]
            self.losses = checkpoint["loss_objects"]
            self.metrics = checkpoint["metric_objects"]
            self.loss_information = checkpoint["loss_information"]
            (self.logger.get_loss_viz()).logs = checkpoint["loss_logs"]
            for load_item in self.model_names + self.optimizer_names:
                getattr(self, load_item).load_state_dict(checkpoint[load_item])
            if load_items is not None:
                if type(load_items) is list:
                    for itms in load_items:
                        setattr(self, itms, checkpoint["itms"])
                else:
                    setattr(self, load_items, checkpoint["load_items"])
        except:
            raise Exception("Model could not be loaded from {}.".format(load_path))

    def _get_argument_maps(self, default_map, func):
        r"""Extracts the signature of the `func`. Then it returns the list of arguments that
        are present in the object and need to be mapped and passed to the `func` when calling it.

        Args:
            default_map (dict): The keys of this dictionary override the function signature.
            func (function): Function whose argument map is to be generated.

        Returns:
            List of arguments that need to be fed into the function. It contains all the positional
            arguments and keyword arguments that are stored in the object. If any of the required
            arguments are not present an error is thrown.
        """
        sig = signature(func)
        arg_map = {}
        for sig_param in sig.parameters.values():
            arg = sig_param.name
            arg_name = arg
            if arg in default_map:
                arg_name = default_map[arg]
            if sig_param.default is not _empty:
                if arg_name in self.__dict__:
                    arg_map.update({arg: arg_name})
            else:
                if arg_name not in self.__dict__ and arg != "kwargs" and arg != "args":
                    raise Exception("Argument : {} not present.".format(arg_name))
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
            args[key] = self.__dict__[val]
        return args

    def train_iter_custom(self):
        r"""Function that needs to be extended if ``train_iter`` is to be modified. Use this function
        to perform any sort of initialization that need to be done at the beginning of any train
        iteration. Refer the model zoo and tutorials for more details on how to write this function.
        """
        pass

    # TODO(avik-pal): Clean up this function and avoid returning values
    def train_iter(self):
        r"""Calls the train_ops of the loss functions. This is the core function of the Trainer. In most
        cases you will never have the need to extend this function. In extreme cases simply extend
        ``train_iter_custom``.

        .. warning::
            This function is needed in this exact state for the Trainer to work correctly. So it is
            highly recommended that this function is not changed even if the ``Trainer`` is subclassed.

        Returns:
            An NTuple of the ``generator loss``, ``discriminator loss``, ``number of times the generator
            was trained`` and the ``number of times the discriminator was trained``.
        """
        self.train_iter_custom()
        ldis, lgen, dis_iter, gen_iter = 0.0, 0.0, 0, 0
        loss_logs = self.logger.get_loss_viz()
        grad_logs = self.logger.get_grad_viz()
        for name, loss in self.losses.items():
            if isinstance(loss, GeneratorLoss) and isinstance(loss, DiscriminatorLoss):
                # NOTE(avik-pal): In most cases this loss is meant to optimize the Discriminator
                #                 but we might need to think of a better solution
                if self.loss_information["generator_iters"] % self.ngen == 0:
                    cur_loss = loss.train_ops(
                        **self._get_arguments(self.loss_arg_maps[name])
                    )
                    loss_logs.logs[name].append(cur_loss)
                    if type(cur_loss) is tuple:
                        lgen, ldis, gen_iter, dis_iter = (
                            lgen + cur_loss[0],
                            ldis + cur_loss[1],
                            gen_iter + 1,
                            dis_iter + 1,
                        )
                    else:
                        # NOTE(avik-pal): We assume that it is a Discriminator Loss by default.
                        ldis, dis_iter = ldis + cur_loss, dis_iter + 1
                for model_name in self.model_names:
                    grad_logs.update_grads(model_name, getattr(self, model_name))
            elif isinstance(loss, GeneratorLoss):
                if self.loss_information["discriminator_iters"] % self.ncritic == 0:
                    cur_loss = loss.train_ops(
                        **self._get_arguments(self.loss_arg_maps[name])
                    )
                    loss_logs.logs[name].append(cur_loss)
                    lgen, gen_iter = lgen + cur_loss, gen_iter + 1
                for model_name in self.model_names:
                    model = getattr(self, model_name)
                    if isinstance(model, Generator):
                        grad_logs.update_grads(model_name, model)
            elif isinstance(loss, DiscriminatorLoss):
                if self.loss_information["generator_iters"] % self.ngen == 0:
                    cur_loss = loss.train_ops(
                        **self._get_arguments(self.loss_arg_maps[name])
                    )
                    loss_logs.logs[name].append(cur_loss)
                    ldis, dis_iter = ldis + cur_loss, dis_iter + 1
                for model_name in self.model_names:
                    model = getattr(self, model_name)
                    if isinstance(model, Discriminator):
                        grad_logs.update_grads(model_name, model)
        return lgen, ldis, gen_iter, dis_iter

    def eval_ops(self, **kwargs):
        r"""Runs all evaluation operations at the end of every epoch. It calls all the metric functions
        that are passed to the Trainer.
        """
        if self.metrics is not None:
            for name, metric in self.metrics.items():
                metric_logs = self.logger.get_metric_viz()
                metric_logs.logs[name].append(
                    metric.metric_ops(**self._get_arguments(self.metric_arg_maps[name]))
                )

    def optim_ops(self):
        r"""Runs all the schedulers at the end of every epoch.
        """
        for scheduler in self.schedulers:
            scheduler.step()

    def train(self, data_loader, **kwargs):
        r"""Uses the information passed by the user while creating the object and trains the model.
        It iterates over the epochs and the DataLoader and calls the functions for training the models
        and logging the required variables.

        .. note::
            Even though ``__call__`` calls this function, it is best if ``train`` is not called directly.
            When ``__call__`` is invoked, we infer the ``batch_size`` from the ``data_loader``. Also,
            we are certain not going to change the interface of the ``__call__`` function so it gives
            the user a stable API, while we can change the flow of execution of ``train`` in future.

        .. warning::
            The user should never try to change this function in subclass. It is too delicate and
            changing affects every other function present in this ``Trainer`` class.

        This function controls the execution of all the components of the ``Trainer``. It controls the
        ``logger``, ``train_iter``, ``save_model``, ``eval_ops`` and ``optim_ops``.

        Args:
            data_loader (torch.utils.data.DataLoader): A DataLoader for the trainer to iterate over
                and train the models.
        """
        for name in self.optimizer_names:
            getattr(self, name).zero_grad()

        for epoch in range(self.start_epoch, self.epochs):

            start_time = time.time()

            for model in self.model_names:
                getattr(self, model).train()

            for data in data_loader:

                if type(data) is tuple or type(data) is list:
                    self.real_inputs = data[0].to(self.device)
                    self.labels = data[1].to(self.device)
                elif type(data) is torch.Tensor:
                    self.real_inputs = data.to(self.device)
                else:
                    self.real_inputs = data

                lgen, ldis, gen_iter, dis_iter = self.train_iter()
                self.loss_information["generator_losses"] += lgen
                self.loss_information["discriminator_losses"] += ldis
                self.loss_information["generator_iters"] += gen_iter
                self.loss_information["discriminator_iters"] += dis_iter

                self.logger.run_mid_epoch(self)

            if "save_items" in kwargs:
                self.save_model(epoch, kwargs["save_items"])
            else:
                self.save_model(epoch)

            for model in self.model_names:
                getattr(self, model).eval()

            self.eval_ops(**kwargs)
            self.logger.run_end_epoch(self, epoch, time.time() - start_time)
            self.optim_ops()

        print("Training of the Model is Complete")

    def complete(self, **kwargs):
        r"""Marks the end of training. It saves the final model and turns off the logger.

        .. note::
            It is not necessary to call this function. If it is not called the logger is kept
            alive in the background. So it might be considered a good practice to call this
            function.
        """
        if "save_items" in kwargs:
            self.save_model(-1, kwargs["save_items"])
        else:
            self.save_model(-1)
        self.logger.close()

    def __call__(self, data_loader, **kwargs):
        self.batch_size = data_loader.batch_size
        self.train(data_loader, **kwargs)
