import torch
import torch.nn.functional as F
from .metric import EvaluationMetric
from ..utils import reduce

__all__ = ['FrechetDistance']


class FrechetDistance(EvaluationMetric):
    r"""
    Computes the Frechet Classifier Distance between a batch of real and generated samples

    Args:
        classifier (torch.nn.Module, optional) : The model to be used as a base to compute the classifier
            score. Apart from the features of the final layer, the model must also return activations from
            and intermediate layer as a second parameter, with respect to which the Frechet Distance is to
            be calculated
        transform (torchvision.transforms, optional) : Transformations applied to the image before feeding
            it to the classifier
    """
    def __init__(self, classifier=None, transform=None):
        # TODO(Aniket1998): Add an approach to extract the correct features from torchvision's Inception
        # model such that the Frechet Classifier Distance defaults to calculating the Frechet Inception distance
        self.classifier = classifier
        self.classifier.eval()
        self.transform = transform

    def preprocess(self, x):
        r"""
        Preprocessor for the Classifier Score. It transforms the image as per the transform requirements
        and feeds it to the classifier.

        Args:
            x (torch.Tensor) : A tuple of real and generated images (x_real, x_generated) in tensor format

        Returns:
            The intermediate activations from the classifier.
        """
        x_real, x_gen = x
        _, activations_real = self.classifier(x_real) if self.transform is None else self.classifier(
            self.transform(x_real))
        _, activations_gen = self.classifier(x_gen) if self.transform is None else self.classifier(
            self.transform(x_gen))
        return activations_real, activations_gen

    def calculate_score(self, x):
        r"""
        Computes the Inception Score for the Input.

        Args:
            x (torch.Tensor) : Image in tensor format

        Returns:
            The Inception Score.
        """
        x_real, x_gen = x
        meansq = F.mse_loss(torch.mean(x_real, dim=1), torch.mean(x_gen, dim=1))

        return torch.exp(reduce(kl, 'elementwise_mean'))
