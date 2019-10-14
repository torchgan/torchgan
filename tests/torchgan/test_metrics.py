import os
import sys
import unittest

import torch
from torchgan.metrics import *

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))


class TestMetrics(unittest.TestCase):
    def test_inception_score(self):
        inception_score = ClassifierScore()
        x = torch.Tensor([[1.0, 2.0, 3.0], [-1.0, 5.0, 3.1]])
        self.assertAlmostEqual(inception_score.calculate_score(x).item(), 1.24357, 4)
