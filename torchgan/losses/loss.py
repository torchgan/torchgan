import torch
import torch.nn as nn

__all__ = ['GeneratorLoss', 'DiscriminatorLoss']

class GeneratorLoss(nn.Module):
    r"""Base class for all generator losses

    Args:
        reduction (string, optional): Specifies the reduction to apply to the output.
        If `none` no reduction will be applied. If `elementwise_mean` the sum of
            the elements will be divided by the number of elements in the output. If
            `sum` the output will be summed.
    """
    def __init__(self, reduction='elementwise_mean', override_train_ops=None):
        super(GeneratorLoss, self).__init__()
        self.reduction = reduction
        self.override_train_ops = override_train_ops

    def train_ops(self, generator, discriminator, optimizer_generator, device, batch_size, labels=None):
        if self.override_train_ops is not None:
            return self.override_train_ops(generator, discriminator, optimizer_generator, device, batch_size, labels)
        else:
            if labels is None and generator.label_type == 'required':
                raise Exception('GAN model requires labels for training')
            noise = torch.randn(batch_size, generator.encoding_dims, device=device)
            optimizer_generator.zero_grad()
            if generator.label_type == 'generated':
                label_gen = torch.randint(0, generator.num_classes, (batch_size,), device=device)
            if generator.label_type == 'none':
                fake = generator(noise)
            elif generator.label_type == 'required':
                fake = generator(noise, labels)
            elif generator.label_type == 'generated':
                fake = generator(noise, label_gen)
            if discriminator.label_type == 'none':
                dgz = discriminator(fake)
            else:
                if generator.label_type == 'generated':
                    dgz = discriminator(fake, label_gen)
                else:
                    dgz = discriminator(fake, labels)
            loss = self.forward(dgz)
            loss.backward()
            optimizer_generator.step()
            return loss.item()

class DiscriminatorLoss(nn.Module):
    r"""Base class for all discriminator losses

    Args:
        reduction (string, optional): Specifies the reduction to apply to the output.
            If `none` no reduction will be applied. If `elementwise_mean` the sum of
            the elements will be divided by the number of elements in the output. If
            `sum` the output will be summed.
    """
    def __init__(self, reduction='elementwise_mean', override_train_ops=None):
        super(DiscriminatorLoss, self).__init__()
        self.reduction = reduction
        self.override_train_ops = override_train_ops

    # NOTE(avik-pal): batch_size and device gets flipped if the order is not given as below. Investigate this
    #                 error as might affect our support for custom loss functions.
    def train_ops(self, generator, discriminator, optimizer_discriminator, real_inputs, device, batch_size,
                  labels=None):
        if self.override_train_ops is not None:
            return self.override_train_ops(self, generator, discriminator, optimizer_discriminator,
                   real_inputs, batch_size, device, labels)
        else:
            if labels is None and (generator.label_type == 'required' or discriminator.label_type == 'required'):
                raise Exception('GAN model requires labels for training')
            noise = torch.randn(real_inputs.size(0), generator.encoding_dims, device=device)
            if generator.label_type == 'generated':
                label_gen = torch.randint(0, generator.num_classes, (real_inputs.size(0),), device=device)
            optimizer_discriminator.zero_grad()
            if discriminator.label_type == 'none':
                dx = discriminator(real_inputs)
            elif discriminator.label_type == 'required':
                dx = discriminator(real_inputs, labels)
            else:
                dx = discriminator(real_inputs, label_gen)
            if generator.label_type == 'none':
                fake = generator(noise)
            elif generator.label_type == 'required':
                fake = generator(noise, labels)
            else:
                fake = generator(noise, label_gen)
            if discriminator.label_type == 'none':
                dgz = discriminator(fake.detach())
            else:
                if generator.label_type == 'generated':
                    dgz = discriminator(fake.detach(), label_gen)
                else:
                    dgz = discriminator(fake.detach(), labels)
            loss = self.forward(dx, dgz)
            loss.backward()
            optimizer_discriminator.step()
            return loss.item()
