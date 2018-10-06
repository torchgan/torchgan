import torch
import torch.data as data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as dsets

__all__ = ['Datasets', 'CIFAR10', 'CIFAR100',
           'MNIST', 'FashionMNIST', 'EMNIST']

class Datasets(object):
    def __init__(self, input_shape, root=".", batch_size=128, output_shape=None):
        self.root = root
        self.batch_size = batch_size
        pad = None
        crop_size = None
        if output_shape == None:
            pad = (0, 0)
        else:
            if output_shape[0] < input_shape[0] or output_shape[1] < input_shape[1]:
                crop_size = output_shape
            else:
                pad = (output_shape[0] - input_shape[0], output_shape[1] - input_shape[1])
        self.transformation = [transforms.ToTensor()]
        if self.pad == None:
            if self.crop_size is not None:
                self.transformation.append(transforms.CenterCrop(self.crop_size))
        else:
            self.transformation.append(transforms.Pad(self.pad))
        self.transformation.append(transforms.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0,1.0,1.0)))

class CIFAR10(Datasets):
    def __init__(self, download=None, **kwargs):
        super(self, CIFAR10).__init__((32, 32), **kwargs)
        self.download = download if download is not None else False

    def __call__(self, train=True, shuffle=True):
        train_dataset = dsets.CIFAR10(root=self.root, train=train,
                                transforms.Compose(self.transformation),
                                download=self.download)
        train_loader = data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=shuffle)

class CIFAR100(Datasets):
    def __init__(self, download=None, **kwargs):
        super(self, CIFAR100).__init__((32, 32), **kwargs)
        self.download = download if download is not None else False

    def __call__(self, train=True, shuffle=True):
        train_dataset = dsets.CIFAR100(root=self.root, train=train,
                                transforms.Compose(self.transformation),
                                download=self.download)
        train_loader = data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=shuffle)

class MNIST(Datasets):
    def __init__(self, download=None, **kwargs):
        super(self, MNIST).__init__((28, 28), **kwargs)
        self.download = download if download is not None else False

    def __call__(self, train=True, shuffle=True):
        train_dataset = dsets.MNIST(root=self.root, train=train,
                                transforms.Compose(self.transformation),
                                download=self.download)
        train_loader = data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=shuffle)

class FashionMNIST(Datasets):
    def __init__(self, download=None, **kwargs):
        super(self, FashionMNIST).__init__((28, 28), **kwargs)
        self.download = download if download is not None else False

    def __call__(self, train=True, shuffle=True):
        train_dataset = dsets.FashionMNIST(root=self.root, train=train,
                                transforms.Compose(self.transformation),
                                download=self.download)
        train_loader = data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=shuffle)

class EMNIST(Datasets):
    def __init__(self, download=None, **kwargs):
        super(self, EMNIST).__init__((28, 28), **kwargs)
        self.download = download if download is not None else False

    def __call__(self, train=True, shuffle=True):
        train_dataset = dsets.EMNIST(root=self.root, train=train,
                                transforms.Compose(self.transformation),
                                download=self.download)
        train_loader = data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=shuffle)

class LSUN(Datasets):
    def __init__(self, input_shape, download=None, **kwargs):
        super(self, LSUN).__init__(input_shape, **kwargs)
        if download is not None:
            raise Exception("LSUN needs to be manually downloaded")

    def __call__(self, train=True, shuffle=True):
        train_dataset = dsets.LSUN(root=self.root, train=train,
                                transforms.Compose(self.transformation))
        train_loader = data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=shuffle)
