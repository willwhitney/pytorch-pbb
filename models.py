import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td

from truncnormal import trunc_normal_

class Gaussian(object):
    def __init__(self, mu, rho):
        super().__init__()
        self.mu = mu
        self.rho = rho
        # self.normal = td.Normal(0, 1)
        self.normal = td.Normal(torch.zeros_like(self.mu), 
                                torch.ones_like(self.mu))


    @property
    def device(self):
        return self.mu.device

    @property
    def sigma(self):
        # return torch.pow(torch.exp(self.rho), 1/2)
        return F.softplus(self.rho)

    # @profile
    def sample(self, n=1):
        if n == 1:
            # epsilon = self.normal.sample(self.rho.size())
            # epsilon = self.normal.sample()
            epsilon = torch.randn_like(self.mu)
        else:
            # epsilon = self.normal.sample((n, *self.rho.size()))
            # epsilon = self.normal.sample((n,))
            epsilon = torch.randn((n,) + self.mu.shape, device=self.device)
        # epsilon = epsilon.to(self.device)
        return self.mu + self.sigma * epsilon

    def log_prob(self, input):
        return (-math.log(math.sqrt(2 * math.pi))
                - torch.log(self.sigma)
                - ((input - self.mu) ** 2) / (2 * self.sigma ** 2)).sum()


class GaussianLinear(nn.Module):
    def __init__(self, in_features, out_features, init_sigma):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        init_rho = np.log(np.exp(init_sigma) - 1.0)
        # init_rho = np.log(init_sigma**2)

        # Weight parameters
        sigma_weights = 1/np.sqrt(in_features)
        threshold = 2 * sigma_weights
        # self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).normal_(std=1/in_features).clamp_(-2/in_features, 2/in_features))

        mu_tensor = torch.Tensor(out_features, in_features)
        trunc_normal_(mu_tensor, 0, sigma_weights, -threshold, threshold)
        self.weight_mu = nn.Parameter(mu_tensor)
        self.weight_rho = nn.Parameter(torch.ones_like(self.weight_mu) * init_rho)
        self.weight = Gaussian(self.weight_mu, self.weight_rho)

        # Bias parameters
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_rho = nn.Parameter(torch.ones_like(self.bias_mu) * init_rho)
        self.bias = Gaussian(self.bias_mu, self.bias_rho)


    def forward(self, x):
        weight = self.weight.sample()
        bias = self.bias.sample()

        return F.linear(x, weight, bias)


    @property
    def device(self):
        return self.weight.device


    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.weight.normal.to(*args, **kwargs)
        self.bias.normal.to(*args, **kwargs)


class PerSampleGaussianLinear(nn.Module):
    def __init__(self, in_features, out_features, init_sigma):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        init_rho = np.log(np.exp(init_sigma) - 1.0)
        # init_rho = np.log(init_sigma**2)

        # Weight parameters
        sigma_weights = 1/np.sqrt(in_features)
        threshold = 2 * sigma_weights
        # self.weight_mu = nn.Parameter(
            # torch.Tensor(out_features, in_features).normal_(std=1/in_features).clamp_(-2/in_features, 2/in_features))
        mu_tensor = torch.Tensor(out_features, in_features)
        trunc_normal_(mu_tensor, 0, sigma_weights, -threshold, threshold)
        self.weight_mu = nn.Parameter(mu_tensor)
        self.weight_rho = nn.Parameter(torch.ones_like(self.weight_mu) * init_rho)
        self.weight = Gaussian(self.weight_mu, self.weight_rho)

        # Bias parameters
        # self.bias_mu = nn.Parameter(torch.Tensor(out_features).normal_(std=1/in_features).clamp_(-2/in_features, 2/in_features))
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_rho = nn.Parameter(torch.ones_like(self.bias_mu) * init_rho)
        self.bias = Gaussian(self.bias_mu, self.bias_rho)

    # @profile
    def forward(self, x):
        bsize = x.shape[0]
        weight = self.weight.sample(bsize)
        bias = self.bias.sample(bsize)
        return torch.baddbmm(bias.view(bsize, -1, 1), weight, x.view(bsize, -1, 1)).view(bsize, -1)

    @property
    def device(self):
        return self.weight.device


class NoisyNet(nn.Module):
    def __init__(self, init_sigma, clipping='tanh', per_sample=False):
        super().__init__()
        self.clipping = clipping
        if per_sample:
            self.l1 = PerSampleGaussianLinear(784, 600, init_sigma)
            self.l2 = PerSampleGaussianLinear(600, 600, init_sigma)
            self.l3 = PerSampleGaussianLinear(600, 600, init_sigma)
            self.lout = PerSampleGaussianLinear(600, 10, init_sigma)
        else:
            self.l1 = GaussianLinear(784, 600, init_sigma)
            self.l2 = GaussianLinear(600, 600, init_sigma)
            self.l3 = GaussianLinear(600, 600, init_sigma)
            self.lout = GaussianLinear(600, 10, init_sigma)
        
    def output_transform(self, x):
        # lower bound output prob
        if self.clipping == 'relu':
            x = F.relu(x + 10) - 10
        elif self.clipping == 'tanh':
            x = F.tanh(x/6) * 6
        elif self.clipping == 'hard':
            pass
        elif self.clipping == 'none':
            pass
        else:
            assert False
            
        output = F.log_softmax(x, dim=1)
        if self.clipping == 'hard':
            output = torch.clamp(output, -6.907755279, 1)
        return output

    def forward(self, x):
        x = torch.flatten(x, 1, -1)
        
        x = self.l1(x)
        x = F.relu(x)
        x = self.l2(x)
        x = F.relu(x)
        x = self.l3(x)
        x = F.relu(x)
        
        x = self.lout(x)
        return self.output_transform(x)

    @property
    def device(self):
        return next(self.parameters()).device
    
    # THIS IS A BIT HACKY
    def get_means(self):
        return [p for i,p in enumerate(self.parameters()) if i % 2 == 0]
    
    def get_sigmas(self):
        return [torch.pow(torch.exp(p), 1/2) for i,p in enumerate(self.parameters()) if i % 2 == 1]
    
class SmallNoisyNet(NoisyNet):
    def __init__(self, init_sigma, per_sample=False, *args, **kwargs):
        super().__init__(init_sigma, per_sample=per_sample, *args, **kwargs)
        if per_sample:
            self.l1 = PerSampleGaussianLinear(784, 500, init_sigma)
            self.lout = PerSampleGaussianLinear(500, 10, init_sigma)
        else:
            self.l1 = GaussianLinear(784, 500, init_sigma)
            self.lout = GaussianLinear(500, 10, init_sigma)
        
    # @profile
    def forward(self, x):
        x = torch.flatten(x, 1, -1)
        
        x = self.l1(x)
        x = F.relu(x)
        x = self.lout(x)
        return self.output_transform(x)


class Lagrangian(nn.Module):
    def __init__(self, init_value):
        super().__init__()
        if init_value > 1:
            init_param = init_value
        else:
            init_param = math.log(math.exp(init_value + 1e-5) - 1)

        self.l = nn.Parameter(torch.tensor(init_param).float())

    def forward(self, *args):
        return F.softplus(self.l)


class SupervisedReprNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.repr = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            nn.Flatten(),
            nn.Linear(1600, 784)
        )
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc2 = nn.Linear(784, 10)

    def forward(self, x):
        x = self.repr(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

    @property
    def device(self):
        return next(self.parameters()).device
