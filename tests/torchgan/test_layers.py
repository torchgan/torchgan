import os
import sys
import unittest

import torch
from torchgan.layers import *

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))


class TestLayers(unittest.TestCase):
    def match_layer_output_shape(self, layer, *inputs, output_shape):
        z = layer(*inputs)

        self.assertEqual(z.shape, output_shape)

    def test_residual_block2d(self):
        input = torch.rand(16, 3, 10, 10)

        layer = ResidualBlock2d([3, 16, 32, 3], [3, 3, 1], paddings=[1, 1, 0])

        self.match_layer_output_shape(
            layer, input, output_shape=(16, 3, 10, 10)
        )

        layer = ResidualBlock2d(
            [3, 16, 32, 1],
            [3, 3, 1],
            paddings=[1, 1, 0],
            shortcut=torch.nn.Conv2d(3, 1, 3, padding=1),
        )

        self.match_layer_output_shape(
            layer, input, output_shape=(16, 1, 10, 10)
        )

        layer = ResidualBlock2d([3, 16, 3], [1, 1])

        self.match_layer_output_shape(
            layer, input, output_shape=(16, 3, 10, 10)
        )

    def test_transposed_residula_block2d(self):
        input = torch.rand(16, 3, 10, 10)

        layer = ResidualBlockTranspose2d(
            [3, 16, 32, 3], [3, 3, 1], paddings=[1, 1, 0]
        )

        self.match_layer_output_shape(
            layer, input, output_shape=(16, 3, 10, 10)
        )

        layer = ResidualBlockTranspose2d(
            [3, 16, 32, 1],
            [3, 3, 1],
            paddings=[1, 1, 0],
            shortcut=torch.nn.Conv2d(3, 1, 3, padding=1),
        )

        self.match_layer_output_shape(
            layer, input, output_shape=(16, 1, 10, 10)
        )

        layer = ResidualBlockTranspose2d([3, 16, 3], [1, 1])

        self.match_layer_output_shape(
            layer, input, output_shape=(16, 3, 10, 10)
        )

    def test_basic_block2d(self):
        input = torch.rand(16, 3, 10, 10)

        layer = BasicBlock2d(3, 13, 3, 1, 1)

        self.match_layer_output_shape(
            layer, input, output_shape=(16, 16, 10, 10)
        )

        layer = BasicBlock2d(3, 13, 3, 1, 1, batchnorm=False)

        self.match_layer_output_shape(
            layer, input, output_shape=(16, 16, 10, 10)
        )

    def test_bottleneck_block2d(self):
        input = torch.rand(16, 3, 10, 10)

        layer = BottleneckBlock2d(3, 13, 3, 1, 1)

        self.match_layer_output_shape(
            layer, input, output_shape=(16, 16, 10, 10)
        )

        layer = BottleneckBlock2d(3, 13, 3, 1, 1, batchnorm=False)

        self.match_layer_output_shape(
            layer, input, output_shape=(16, 16, 10, 10)
        )

    def test_transition_block2d(self):
        input = torch.rand(16, 3, 10, 10)

        layer = TransitionBlock2d(3, 16, 3, 1, 1)

        self.match_layer_output_shape(
            layer, input, output_shape=(16, 16, 10, 10)
        )

        layer = TransitionBlock2d(3, 16, 3, 1, 1, batchnorm=False)

        self.match_layer_output_shape(
            layer, input, output_shape=(16, 16, 10, 10)
        )

    def test_transition_block_transpose2d(self):
        input = torch.rand(16, 3, 10, 10)

        layer = TransitionBlockTranspose2d(3, 16, 3, 1, 1)

        self.match_layer_output_shape(
            layer, input, output_shape=(16, 16, 10, 10)
        )

        layer = TransitionBlockTranspose2d(3, 16, 3, 1, 1, batchnorm=False)

        self.match_layer_output_shape(
            layer, input, output_shape=(16, 16, 10, 10)
        )

    def test_dense_block2d(self):
        input = torch.rand(16, 3, 10, 10)

        layer = DenseBlock2d(5, 3, 16, BottleneckBlock2d, 3, padding=1)

        self.match_layer_output_shape(
            layer, input, output_shape=(16, 83, 10, 10)
        )

    def test_self_attention2d(self):
        input = torch.rand(16, 88, 10, 10)

        layer = SelfAttention2d(88)

        self.match_layer_output_shape(
            layer, input, output_shape=(16, 88, 10, 10)
        )

    def test_spectral_norm2d(self):
        input = torch.rand(16, 3, 10, 10)

        layer = SpectralNorm2d(
            torch.nn.Conv2d(3, 10, 3, padding=1), power_iterations=10
        )

        self.match_layer_output_shape(
            layer, input, output_shape=(16, 10, 10, 10)
        )

    def test_virtual_batchnorm(self):
        reference_batch = torch.rand(16, 3, 10, 10)
        input = torch.rand(16, 3, 10, 10)

        layer = VirtualBatchNorm(3)
        layer(reference_batch)

        self.match_layer_output_shape(
            layer, input, False, output_shape=(16, 3, 10, 10)
        )

        assert layer.ref_mu is not None and layer.ref_var is not None

        self.match_layer_output_shape(
            layer, input, True, output_shape=(16, 3, 10, 10)
        )

        assert layer.ref_mu is None and layer.ref_var is None

    def test_minibatch_discrimination(self):
        input = torch.rand(16, 100)

        layer = MinibatchDiscrimination1d(100, 200)

        self.match_layer_output_shape(
            layer, input, output_shape=(16, 100 + 200)
        )
