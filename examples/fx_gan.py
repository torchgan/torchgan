import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch, pdb
import torch.nn as nn
import torch.autograd
from torch.autograd import Variable
import torch.utils.data

from sklearn.preprocessing import QuantileTransformer

feat_length = 168
input_size = 120
feat_channel = 10
epochs = 100

class out_layer(nn.Module):
    def __init__(self, in_channels, out_channels = 1, gap_size = feat_length):
        super(out_layer, self).__init__()
        self.in_channels = in_channels
        self.gap = nn.AvgPool1d(kernel_size = gap_size, padding = 1)
        self.lin = nn.Linear(in_channels, out_channels)
        self.activation = nn.Sigmoid()
    def forward(self, x):
        gap = self.gap(x).view(-1, self.in_channels)
        return self.activation(self.lin(gap))

class base_conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, ks = 3, pd = 1, do = 0.0, **kwargs):
        super(base_conv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, 
            bias = False, kernel_size = ks, padding=pd, **kwargs)
        self.activation = nn.RReLU(inplace = True)
        self.conv.weight.data.normal_(0, 0.01)
    def forward(self, x):
        return self.activation(self.conv(x))

# Discriminator needs to take the generator output or the real sample and guess whether fake or not
class Discriminator(nn.Module):
    def __init__(self, channels = 1, out_channels = 1, gap_size = feat_length, noise = 0.1):
        super(Discriminator, self).__init__()
        self.noise = noise
        self.conv1 = base_conv1d(in_channels = channels, out_channels = 512, ks = 7, pd = 3)
        self.conv2 = base_conv1d(in_channels = 512, out_channels = 256, ks = 5, pd = 2)
        self.conv3 = base_conv1d(in_channels = 256, out_channels = 128)
        self.out = out_layer(in_channels = 128, out_channels = out_channels, gap_size = gap_size)
    def forward(self, x):
        if self.training and self.noise != 0.0:
            x = x + Variable(x.data.new(x.size()).normal_(0.0, self.noise))
        return self.out(self.conv3(self.conv2(self.conv1(x))))

# Generator needs to take the latent variables and then generate a fake sequence of given channels
class Generator(nn.Module):
    def __init__(self, channels = 1, gap_size = feat_length):
        super(Generator, self).__init__()
        self.channels = channels
        self.gap_size = gap_size
        self.conv1 = base_conv1d(in_channels = 128, out_channels = 256)
        self.conv2 = base_conv1d(in_channels = 256, out_channels = 512, ks = 5, pd = 2)
        self.conv3 = base_conv1d(in_channels = 512, out_channels = feat_length-input_size, ks = 7, pd = 3)
        self.conv3.activation = nn.Sigmoid() # Because the input is in [0, 1]
    def forward(self, x, direction = True):
        gen = Variable(torch.randn(x.shape[0], 128, self.channels)).cuda()
        gen = self.conv3(self.conv2(self.conv1(gen))).view(-1, self.channels, feat_length-input_size)
        return torch.cat([gen, x], dim=2) if direction else torch.cat([x, gen], dim=2)

Dataset = pd.read_csv('Dataset.csv', index_col='Gmt time', parse_dates=True, infer_datetime_format=True)
Dataset = Dataset.drop(['Imputated'], axis=1)

input_vectors = []

for i in range(len(Dataset)//feat_length):
    start = i * feat_length
    end = start + input_size
    input_vector = Dataset.iloc[start:end].values
    input_vector = np.swapaxes(input_vector, 0, 1)
    input_vectors.append(input_vector)

X = np.asarray(input_vectors)

X_transformer = QuantileTransformer(output_distribution='uniform')
X = X_transformer.fit_transform(X.reshape(-1, input_size))
X_train = torch.from_numpy(X.reshape(-1, feat_channel, input_size))

train_dataset = torch.utils.data.TensorDataset(X_train)

train_loader = torch.utils.data.DataLoader(train_dataset, 
    batch_size = 64, shuffle = True, pin_memory = True)

G = Generator(channels = feat_channel, gap_size = feat_length).cuda()
D = Discriminator(channels = feat_channel, gap_size = feat_length).cuda()

G_opt = torch.optim.Adam(params = G.parameters(), lr = 1e-3)
D_opt = torch.optim.Adam(params = D.parameters(), lr = 1e-3)

BCE = torch.nn.BCELoss()

G.train(True)
D.train(True)

for epoch in range(epochs):

    for i, features in enumerate(train_loader):

        G.zero_grad()
        D.zero_grad()
        torch.cuda.empty_cache()

        real_features = features[0].float().cuda(non_blocking=True)
        real_features = real_features.view(-1, feat_channel, input_size)

        # Q:shall we divide them two to make sure
        # only one side of a specific signal used
        half = real_features.shape[0]//2
        features_one = real_features[:half,:,:]
        features_two = real_features[half:,:,:]

        fake_true = D(G(features_one, True))
        fake_false = D(G(features_two, False))

        T_loss = BCE(fake_true, 
            torch.ones(fake_true.shape[0]).cuda())
        F_loss = BCE(fake_false, 
            torch.ones(fake_false.shape[0]).cuda())
        
        if torch.rand(1) > 0.5:
            #GAN_loss = T_loss - F_loss
            GAN_loss = fake_true.mean() - fake_false.mean()
            D_opt.zero_grad()
            GAN_loss.backward()
            D_opt.step()
        else:
            #GAN_loss = F_loss - T_loss
            GAN_loss = fake_false.mean() - fake_true.mean()
            G_opt.zero_grad()
            GAN_loss.backward()
            G_opt.step()

        P = fake_true.mean()
        R = 1-fake_false.mean()
        print('%f %f' % (GAN_loss.abs(), 4.0*P*R/(P+R)))