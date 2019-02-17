import torch

from .functional import auxiliary_classification_loss
from .loss import DiscriminatorLoss, GeneratorLoss

__all__ = ["AuxiliaryClassifierGeneratorLoss", "AuxiliaryClassifierDiscriminatorLoss"]


class AuxiliaryClassifierGeneratorLoss(GeneratorLoss):
    r"""Auxiliary Classifier GAN (ACGAN) loss based on a from
    `"Conditional Image Synthesis With Auxiliary Classifier GANs
    by Odena et. al. " <https://arxiv.org/abs/1610.09585>`_ paper

    Args:
       reduction (str, optional): Specifies the reduction to apply to the output.
            If ``none`` no reduction will be applied. If ``mean`` the outputs are averaged over batch size.
            If ``sum`` the elements of the output are summed.
        override_train_ops (function, optional): A function is passed to this argument,
            if the default ``train_ops`` is not to be used.
    """

    def forward(self, logits, labels):
        return auxiliary_classification_loss(logits, labels, self.reduction)

    def train_ops(
        self,
        generator,
        discriminator,
        optimizer_generator,
        device,
        batch_size,
        labels=None,
    ):
        r"""Defines the standard ``train_ops`` used by the Auxiliary Classifier generator loss.

        The ``standard optimization algorithm`` for the ``discriminator`` defined in this train_ops
        is as follows (label_g and label_d both could be either real labels or generated labels):

        1. :math:`fake = generator(noise, label_g)`
        2. :math:`value_1 = classifier(fake, label_g)`
        3. :math:`value_2 = classifier(real, label_d)`
        4. :math:`loss = loss\_function(value_1, label_g) + loss\_function(value_2, label_d)`
        5. Backpropagate by computing :math:`\nabla loss`
        6. Run a step of the optimizer for discriminator

        Args:
            generator (torchgan.models.Generator): The model to be optimized. For ACGAN, it must require
                                                   labels for training
            discriminator (torchgan.models.Discriminator): The discriminator which judges the
                performance of the generator.
            optimizer_discriminator (torch.optim.Optimizer): Optimizer which updates the ``parameters``
                of the ``discriminator``.
            real_inputs (torch.Tensor): The real data to be fed to the ``discriminator``.
            device (torch.device): Device on which the ``generator`` and ``discriminator`` is present.
            batch_size (int): Batch Size of the data infered from the ``DataLoader`` by the ``Trainer``.
            labels (torch.Tensor, optional): Labels for the data.

        Returns:
            Scalar value of the loss.
        """
        if self.override_train_ops is not None:
            return self.override_train_ops(
                generator,
                discriminator,
                optimizer_generator,
                device,
                batch_size,
                labels,
            )
        if generator.label_type == "required" and labels is None:
            raise Exception("GAN model requires label for training")
        noise = torch.randn(batch_size, generator.encoding_dims, device=device)
        optimizer_generator.zero_grad()
        if generator.label_type == "none":
            raise Exception("Incorrect Model: ACGAN generator must require labels")
        if generator.label_type == "required":
            fake = generator(noise, labels)
        elif generator.label_type == "generated":
            label_gen = torch.randint(
                0, generator.num_classes, (batch_size,), device=device
            )
            fake = generator(noise, label_gen)
        cgz = discriminator(fake, mode="classifier")
        if generator.label_type == "required":
            loss = self.forward(cgz, labels)
        else:
            label_gen = label_gen.type(torch.LongTensor).to(device)
            loss = self.forward(cgz, label_gen)
        loss.backward()
        optimizer_generator.step()
        return loss.item()


class AuxiliaryClassifierDiscriminatorLoss(DiscriminatorLoss):
    r"""Auxiliary Classifier GAN (ACGAN) loss based on a from
    `"Conditional Image Synthesis With Auxiliary Classifier GANs
    by Odena et. al. " <https://arxiv.org/abs/1610.09585>`_ paper

    Args:
       reduction (str, optional): Specifies the reduction to apply to the output.
            If ``none`` no reduction will be applied. If ``mean`` the outputs are averaged over batch size.
            If ``sum`` the elements of the output are summed.
       override_train_ops (function, optional): A function is passed to this argument,
            if the default ``train_ops`` is not to be used.
    """

    def forward(self, logits, labels):
        return auxiliary_classification_loss(logits, labels, self.reduction)

    def train_ops(
        self,
        generator,
        discriminator,
        optimizer_discriminator,
        real_inputs,
        device,
        labels=None,
    ):
        r"""Defines the standard ``train_ops`` used by the Auxiliary Classifier discriminator loss.

        The ``standard optimization algorithm`` for the ``discriminator`` defined in this train_ops
        is as follows (label_g and label_d both could be either real labels or generated labels):

        1. :math:`fake = generator(noise, label_g)`
        2. :math:`value_1 = classifier(fake, label_g)`
        3. :math:`value_2 = classifier(real, label_d)`
        4. :math:`loss = loss\_function(value_1, label_g) + loss\_function(value_2, label_d)`
        5. Backpropagate by computing :math:`\nabla loss`
        6. Run a step of the optimizer for discriminator

        Args:
            generator (torchgan.models.Generator): The model to be optimized. For ACGAN, it must require labels
                                                   for training
            discriminator (torchgan.models.Discriminator): The discriminator which judges the
                performance of the generator.
            optimizer_discriminator (torch.optim.Optimizer): Optimizer which updates the ``parameters``
                of the ``discriminator``.
            real_inputs (torch.Tensor): The real data to be fed to the ``discriminator``.
            device (torch.device): Device on which the ``generator`` and ``discriminator`` is present.
            batch_size (int): Batch Size of the data infered from the ``DataLoader`` by the ``Trainer``.
            labels (torch.Tensor, optional): Labels for the data.

        Returns:
            Scalar value of the loss.
        """
        if self.override_train_ops is not None:
            return self.override_train_ops(
                generator,
                discriminator,
                optimizer_discriminator,
                real_inputs,
                device,
                labels,
            )
        if labels is None:
            raise Exception("ACGAN Discriminator requires labels for training")
        if generator.label_type == "none":
            raise Exception(
                "Incorrect Model: ACGAN generator must require labels for training"
            )
        batch_size = real_inputs.size(0)
        noise = torch.randn(batch_size, generator.encoding_dims, device=device)
        optimizer_discriminator.zero_grad()
        cx = discriminator(real_inputs, mode="classifier")
        if generator.label_type == "required":
            fake = generator(noise, labels)
        elif generator.label_type == "generated":
            label_gen = torch.randint(
                0, generator.num_classes, (batch_size,), device=device
            )
            fake = generator(noise, label_gen)
        cgz = discriminator(fake, mode="classifier")
        if generator.label_type == "required":
            loss = self.forward(cgz, labels) + self.forward(cx, labels)
        else:
            label_gen = label_gen.type(torch.LongTensor).to(device)
            loss = self.forward(cgz, label_gen) + self.forward(cx, labels)
        loss.backward()
        optimizer_discriminator.step()
        return loss.item()
