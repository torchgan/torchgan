import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as dsets

__all__ = ['Datasets', 'CIFAR10', 'CIFAR100',
           'MNIST', 'FashionMNIST', 'EMNIST',
           'LSUN']

class Datasets(object):
    def __init__(self, input_shape, root=".", batch_size=128, output_shape=None):
        self.root = root
        self.batch_size = batch_size
        pad = None
        crop_size = None
        if output_shape is None:
            pad = (0, 0)
        else:
            if output_shape[0] < input_shape[0] or output_shape[1] < input_shape[1]:
                crop_size = output_shape
            else:
                pad = (output_shape[0] - input_shape[0], output_shape[1] - input_shape[1])
        self.transformation = [transforms.ToTensor()]
        if pad is None:
            if crop_size is not None:
                self.transformation.append(transforms.CenterCrop(crop_size))
        else:
            self.transformation.append(transforms.Pad(pad))
        self.transformation.append(transforms.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)))

    def transform_update(self, transformation, pos=-1):
        self.transformation.insert(transformation, pos)

class CIFAR10(Datasets):
    def __init__(self, download=None, **kwargs):
        super(CIFAR10, self).__init__((32, 32), **kwargs)
        self.download = download if download is not None else True

    def __call__(self, train=True, shuffle=True):
        train_dataset = dsets.CIFAR10(root=self.root, train=train,
                                      transform=transforms.Compose(self.transformation),
                                      download=self.download)
        train_loader = data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=shuffle)
        return train_loader

class CIFAR100(Datasets):
    def __init__(self, download=None, **kwargs):
        super(CIFAR100, self).__init__((32, 32), **kwargs)
        self.download = download if download is not None else True

    def __call__(self, train=True, shuffle=True):
        train_dataset = dsets.CIFAR100(root=self.root, train=train,
                                       transform=transforms.Compose(self.transformation),
                                       download=self.download)
        train_loader = data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=shuffle)
        return train_loader

class MNIST(Datasets):
    def __init__(self, download=None, **kwargs):
        super(MNIST, self).__init__((28, 28), **kwargs)
        self.download = download if download is not None else True

    def __call__(self, train=True, shuffle=True):
        train_dataset = dsets.MNIST(root=self.root, train=train,
                                    transform=transforms.Compose(self.transformation),
                                    download=self.download)
        train_loader = data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=shuffle)
        return train_loader

class FashionMNIST(Datasets):
    def __init__(self, download=None, **kwargs):
        super(FashionMNIST, self).__init__((28, 28), **kwargs)
        self.download = download if download is not None else True

    def __call__(self, train=True, shuffle=True):
        train_dataset = dsets.FashionMNIST(root=self.root, train=train,
                                           transform=transforms.Compose(self.transformation),
                                           download=self.download)
        train_loader = data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=shuffle)
        return train_loader

class EMNIST(Datasets):
    def __init__(self, download=None, **kwargs):
        super(EMNIST, self).__init__((28, 28), **kwargs)
        self.download = download if download is not None else True

    def __call__(self, train=True, shuffle=True):
        train_dataset = dsets.EMNIST(root=self.root, train=train,
                                     transform=transforms.Compose(self.transformation),
                                     download=self.download)
        train_loader = data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=shuffle)
        return train_loader

class LSUN(Datasets):
    def __init__(self, input_shape, download=None, **kwargs):
        super(LSUN, self).__init__(input_shape, **kwargs)
        if download is not None:
            raise Exception("LSUN needs to be manually downloaded")

    def __call__(self, train=True, shuffle=True):
        train_dataset = dsets.LSUN(root=self.root, train=train,
                                   transform=transforms.Compose(self.transformation))
        train_loader = data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=shuffle)
        return train_loader
