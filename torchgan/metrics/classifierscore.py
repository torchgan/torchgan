import torch
import torch.nn.functional as F
import torchvision
from .metric import EvaluationMetric
from ..utils import reduce

__all__ = ['ClassifierScore']


class ClassifierScore(EvaluationMetric):
    def __init__(self, classifier=None, transform=None):
        self.classifier = torchvision.models.inception_v3(True) if classifier is None else classifier
        self.classifier.eval()
        self.transform = transform

    def preprocess(self, x):
        return self.classifier(x) if self.transform is None else self.classifier(self.transform(x))

    def calculate_score(self, x):
        p = F.softmax(x, dim=1)
        q = torch.mean(p, dim=0)
        kl = torch.sum(p * (F.log_softmax(x, dim=1) - torch.log(q)), dim=1)
        return torch.exp(reduce(kl, 'elementwise_mean'))
