import torch
import torch.nn as nn
import numpy as np

from args import device
from torch.distributions.uniform import Uniform

class Preprocess(nn.Module):
    """
    preprocess layer

    (not used, maybe future)
    """
    def __init__(self):
        super(Preprocess, self).__init__()

    '''
    def forward(self, x, reverse=False, alpha=0.05):
        #if reverse:
        #    print(x.size())
        device = x.device

        if reverse:
            return torch.clip((torch.sigmoid(x) - alpha) / (1-alpha), min=0, max=1) #RICHTIG
            #return torch.clip(torch.floor(256 / 0.90 * (torch.sigmoid(x) - 0.05)), min=0, max=255) / 255

        x = (alpha + (1-alpha) * x)
        # f(x) = a + (1-a)*x => df/dx = 1-a

        x += Uniform(0.0, 1/256).sample(x.size()).to(device)

        # verhindert 0 division bei logit(x)
        x = torch.clip(x, min=0, max=1-(1/256))

        z = torch.logit(x)
        # f(x) = logit(x) => df/dx = ((1/x)-(-(1/(1-x))))

        # df/dx aufsummiert da log
        log_det_jacobian = ((1/x)-(-(1/(1-x)))).log() + torch.tensor((1-alpha)).log().to(device) #REMOVE /255 ?
        return z, log_det_jacobian

    '''
    def forward(self, x, reverse=False):

        if reverse:
            y2 = torch.sigmoid(x)
            y1 = (y2-0.05)/0.9
            return y1

        x += Uniform(0.0, 1/256).sample(x.size()).to(device)
        y1 = 0.05+0.9*x
        y2 = torch.logit(y1)

        jac = self.calc_jac(y1)
        return y2, jac

    def calc_jac(self, y1):
        y1_step = np.log(0.9)
        y2_step = torch.log(1/(y1*(-y1+1)))
        return y1_step+y2_step

    '''
    def forward(self, x, reverse=False):
        if reverse:
            return x/255
        x = x*255
        x += Uniform(0.0, 1.0).sample(x.size()).to(device)
        log_det_jacobian = torch.zeros_like(x)

        log_det_jacobian -= torch.log(torch.tensor(255))

        return x, log_det_jacobian
    '''

    #def forward(self, x, reverse=False):
    #    if reverse:
    #        #print(x.mean())
    #        #print(x.max())
    #        #print(x.min())
    #        #x = x/255.
    #        return torch.clip(x, min=0, max=1)
#
    #    #x = torch.floor(x * 255.) #[0, 1] -> [0, 255]
    #    #x = x + torch.rand(x.shape).to(device)
    #    #return x.clip(0., 255.), torch.fill(torch.empty_like(x), 255.).to(device)
    #    #return x.clip(0, 255), torch.zeros_like(x).to(device)
#
    #    uniform_noise = Uniform(torch.tensor(0.).to(device), torch.tensor(1./255.).to(device))
#
    #    x = x + uniform_noise.sample(x.shape)
    #    #print(x)
    #    return x.clip(0, 1), torch.zeros_like(x).to(device)


class Preprocess2(nn.Module):
    """
    preprocess2 layer

    (not used, maybe future)
    """
    def __init__(self):
        super(Preprocess2, self).__init__()

    def forward(self, x, reverse=False):
        if reverse:
            return x

        x += Uniform(0.0, 1/256).sample(x.size()).to(device)

        return x.clip(0, 1), torch.zeros_like(x)



class WeightNormConv2d(nn.Module):
    """
    weighted 2D convolution
    Inputs:
        in_channel - amount C_in input image
        out_channel - amount C_out
        kernel_size -  size of the convolving kernel
    """
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True):
        super(WeightNormConv2d, self).__init__()
        self.conv = nn.utils.weight_norm(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding, bias=bias))

    def forward(self, x):
        return self.conv(x)


class LayerNorm(nn.Module):
    """
    layer norm across channels in an image
    Inputs:
        c_in - Number of channels of the input
        eps - Small constant to stabilize std
    """
    def __init__(self, c_in, eps=1e-5):

        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, c_in, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, c_in, 1, 1))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, unbiased=False, keepdim=True)
        y = (x - mean) / torch.sqrt(var + self.eps)
        y = y * self.gamma + self.beta
        return y


class LinearBatchNorm(nn.Module):
    """
    This class is mostly inspired from this one:
    https://github.com/kamenbliznashki/normalizing_flows/blob/master/maf.py

    transformation batch norm
    Inputs:
        input_size - (CxHxW)
        momentum - importance previous moving average
        eps - small constant
    """
    def __init__(self, input_size, momentum=0.9, eps=1e-5):
        super().__init__()
        self.momentum = momentum
        self.eps = eps

        self.log_gamma = nn.Parameter(torch.zeros(input_size))
        self.beta = nn.Parameter(torch.zeros(input_size))

        self.register_buffer('running_mean', torch.zeros(input_size))
        self.register_buffer('running_var', torch.ones(input_size))

    def forward(self, x, reverse=False):
        if not reverse:
            if self.training:
                # mean/var jedes pixel im batch => (mean_1,1 = mean(sum(x_1,1) of all imgs in batch))
                self.batch_mean = x.mean(0)
                self.batch_var = x.var(0)

                # update running mean/var
                self.running_mean.mul_(self.momentum).add_(self.batch_mean.data * (1 - self.momentum))
                self.running_var.mul_(self.momentum).add_(self.batch_var.data * (1 - self.momentum))

                mean = self.batch_mean
                var = self.batch_var
            else:
                # if not training, use running mean/var
                mean = self.running_mean
                var = self.running_var

            x_hat = (x - mean) / torch.sqrt(var + self.eps)
            # f(x) = x-m/sqrt(v+e) => log(df/dx) = -0.5 * log(v+e) = log(1/sqrt(v+e))

            y = self.log_gamma.exp() * x_hat + self.beta
            # f(x) = e^g*x + b => log(df/dx) = g

            log_det = self.log_gamma - 0.5 * torch.log(var + self.eps)
            return y, log_det

        elif reverse:
            mean = self.running_mean
            var = self.running_var

            x_hat = (x - self.beta) * torch.exp(-self.log_gamma)
            x = x_hat * torch.sqrt(var + self.eps) + mean

            # log_det = 0.5 * torch.log(var + self.eps) - self.log_gamma

            y = x
            return y

class RescaleLayer(nn.Module):
    """
    rescale layer for NICE
    Inputs:
        input_size - (CxHxW)
    """
    def __init__(self, input_size):
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros(input_size))

    def forward(self, x, reverse=False):
        if not reverse:
            sign = 1
            z = x * torch.exp(sign*self.log_scale)
            return z, self.log_scale
        elif reverse:
            sign = -1
            z = x * torch.exp(sign*self.log_scale)
            return z