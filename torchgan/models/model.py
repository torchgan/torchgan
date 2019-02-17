import torch
import torch.nn as nn

__all__ = ["Generator", "Discriminator"]


class Generator(nn.Module):
    r"""Base class for all Generator models. All Generator models must subclass this.

    Args:
        encoding_dims (int): Dimensions of the sample from the noise prior.
        label_type (str, optional): The type of labels expected by the Generator. The available
            choices are 'none' if no label is needed, 'required' if the original labels are
            needed and 'generated' if labels are to be sampled from a distribution.
    """
    # FIXME(Aniket1998): If a user is overriding the default initializer, he must also
    # override the constructor. Find an efficient workaround by fixing the initialization mechanism
    def __init__(self, encoding_dims, label_type="none"):
        super(Generator, self).__init__()
        self.encoding_dims = encoding_dims
        self.label_type = label_type

    # TODO(Aniket1998): Think of better dictionary lookup based approaches to initialization
    # That allows easy and customizable weight initialization without overriding
    def _weight_initializer(self):
        r"""Default weight initializer for all generator models.
        Models that require custom weight initialization can override this method
        """
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def sampler(self, sample_size, device):
        r"""Function to allow sampling data at inference time. Models requiring
        input in any other format must override it in the subclass.

        Args:
            sample_size (int): The number of images to be generated
            device (torch.device): The device on which the data must be generated

        Returns:
            A list of the items required as input
        """
        return [torch.randn(sample_size, self.encoding_dims, device=device)]


class Discriminator(nn.Module):
    r"""Base class for all Discriminator models. All Discriminator models must subclass this.

    Args:
        input_dims (int): Dimensions of the input.
        label_type (str, optional): The type of labels expected by the Discriminator. The available
            choices are 'none' if no label is needed, 'required' if the original labels are
            needed and 'generated' if labels are to be sampled from a distribution.
    """

    def __init__(self, input_dims, label_type="none"):
        super(Discriminator, self).__init__()
        self.input_dims = input_dims
        self.label_type = label_type

    # TODO(Aniket1998): Think of better dictionary lookup based approaches to initialization
    # That allows easy and customizable weight initialization without overriding
    def _weight_initializer(self):
        r"""Default weight initializer for all disciminator models.
        Models that require custom weight initialization can override this method
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
