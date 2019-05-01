import torch
import torch.nn.functional as F
import torchvision

from ..utils import reduce
from .metric import EvaluationMetric

__all__ = ["ClassifierScore"]


class ClassifierScore(EvaluationMetric):
    r"""
    Computes the Classifier Score of a Model. Also popularly known as the Inception Score.
    The ``classifier`` can be any model. It also supports models outside of torchvision models.
    For more details on how to use custom trained models look up the tutorials.

    Args:
        classifier (torch.nn.Module, optional) : The model to be used as a base to compute the classifier
            score. If ``None`` is passed the pretrained ``torchvision.models.inception_v3`` is used.

            .. note ::
                Ensure that the classifier is on the same ``device`` as the Trainer to avoid sudden
                crash.
        transform (torchvision.transforms, optional) : Transformations applied to the image before feeding
            it to the classifier. Look up the documentation of the torchvision models for this transforms.
        sample_size (int): Batch Size for calculation of Classifier Score.
    """

    def __init__(self, classifier=None, transform=None, sample_size=1):
        super(ClassifierScore, self).__init__()
        self.classifier = (
            torchvision.models.inception_v3(True) if classifier is None else classifier
        )
        self.classifier.eval()
        self.transform = transform
        self.sample_size = sample_size

    def preprocess(self, x):
        r"""
        Preprocessor for the Classifier Score. It transforms the image as per the transform requirements
        and feeds it to the classifier.

        Args:
            x (torch.Tensor) : Image in tensor format

        Returns:
            The output from the classifier.
        """
        x = x if self.transform is None else self.transform(x)
        return self.classifier(x)

    def calculate_score(self, x):
        r"""
        Computes the Inception Score for the Input.

        Args:
            x (torch.Tensor) : Image in tensor format

        Returns:
            The Inception Score.
        """
        p = F.softmax(x, dim=1)
        q = torch.mean(p, dim=0)
        kl = torch.sum(p * (F.log_softmax(x, dim=1) - torch.log(q)), dim=1)
        return torch.exp(reduce(kl, "mean")).data

    def metric_ops(self, generator, device):
        r"""Defines the set of operations necessary to compute the ClassifierScore.

        Args:
            generator (torchgan.models.Generator): The generator which needs to be evaluated.
            device (torch.device): Device on which the generator is present.

        Returns:
            The Classifier Score (scalar quantity)
        """
        noise = torch.randn(self.sample_size, generator.encoding_dims, device=device)
        img = generator(noise).detach()
        score = self.__call__(img)
        return score
