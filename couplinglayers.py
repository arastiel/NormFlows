import torch
import torch.nn as nn

from couplingnetworks import m_net, EasyNet
from args import device

class CouplingLayer(nn.Module):
    """
    det(jac) = mul(s.exp) => log(det(jac)) = sum(s)

    additive coupling layer (NICE)
    Inputs:
        mask_type - chosen mask type
        input_dim - (CxHxW)
        top_left_zero - 0/1 start mask
        c_in - input features
        c_hidden - hidden features
    """

    def __init__(self, mask_type, input_dim=(1, 28, 28), top_left_zero=0, c_in=1, c_hidden=64):
        super(CouplingLayer, self).__init__()
        self.mask_type = mask_type
        self.top_left_zero = top_left_zero
        self.c_in = c_in
        self.c_hidden = c_hidden

        #hiddenlayers 5, hiddenunits=1000 https://arxiv.org/pdf/1410.8516.pdf
        self.m_net = m_net(c_in=input_dim[0]*input_dim[1]*input_dim[2], c_hidden=c_hidden, num_layers=5)

        #self.m_net = EasyNet(c_in=self.c_in, c_hidden=self.c_hidden)
        #self.s = torch.nn.Parameter(torch.zeros(input_dim), requires_grad=True)
        #self.s = torch.rand(input_dim).to(device)
        #self.scale_scale = nn.Parameter(torch.ones(input_dim), requires_grad=True)
        #self.shift_scale = nn.Parameter(torch.zeros(input_dim), requires_grad=True)

        if self.mask_type == "checkerboard":
            self.mask = self.create_che_mask(input_dim[1], input_dim[2], self.top_left_zero)
        if self.mask_type == "half":
            self.mask = self.half_mask(input_dim, self.top_left_zero)


    def half_mask(self, input_dim, top=0):
        mask = torch.empty(input_dim, device=device)
        if top:
            mask[:, 0:input_dim[1]//2, :] = 1
            mask[:, input_dim[1]//2:input_dim[1], :] = 0
        else:
            mask[:, 0:input_dim[1]//2, :] = 0
            mask[:, input_dim[1]//2:input_dim[1], :] = 1
        return mask

    def create_che_mask(self, height, width, top_left_zero=0):
        mask = (torch.arange(height).view(-1, 1) + torch.arange(width))
        if not top_left_zero:
            mask += 1
        return (mask % 2).unsqueeze(0).unsqueeze(0)  # fakedim

    def forward(self, x, reverse=False):
        #print(x[0])
        self.mask = self.mask.to(x.device)
        x_masked = x * self.mask

        m = self.m_net(x_masked)
        #print(len(self.m_net))
        anti_mask = 1 - self.mask

        m = m * anti_mask

        if reverse:
            x = x - m
            return x

        else:
            x = x + m
            # f(x) = x*e^s + t => log(df/dx) = s
            return x, torch.zeros_like(m)

    #def forward(self, x, reverse=False):
    #    x1 = x[]

class CouplingLayer2(nn.Module):
    """
    det(jac) = mul(s.exp) => log(det(jac)) = sum(s)

    affine coupling layer (RealNVP)
    Inputs:
        mask_type - chosen mask type
        input_dim - (CxHxW)
        top_left_zero - 0/1 start mask
        c_in - input features
        c_hidden - hidden features
    """

    def __init__(self, mask_type, input_dim=(1, 28, 28), top_left_zero=0, c_in=1, c_hidden=64):
        super(CouplingLayer2, self).__init__()
        self.scale_scale = nn.Parameter(torch.ones(input_dim), requires_grad=True)
        self.shift_scale = nn.Parameter(torch.zeros(input_dim), requires_grad=True)
        self.mask_type = mask_type
        self.top_left_zero = top_left_zero
        self.c_in = c_in
        self.c_hidden = c_hidden

        self.s_net = EasyNet(c_in=self.c_in, c_hidden=self.c_hidden)
        self.t_net = EasyNet(c_in=self.c_in, c_hidden=self.c_hidden)

        if self.mask_type == "checkerboard":
            self.mask = self.create_che_mask(input_dim[1], input_dim[2], self.top_left_zero)

        elif self.mask_type == "channel":
            self.mask = self.create_cha_mask(input_dim, self.top_left_zero)

    def create_che_mask(self, height, width, top_left_zero=0):
        mask = (torch.arange(height).view(-1, 1) + torch.arange(width))
        if not top_left_zero:
            mask += 1
        return (mask % 2).unsqueeze(0).unsqueeze(0)  # fakedim

    def create_cha_mask(self, input_dim, rev=0):
        if rev:
            mask = torch.cat([torch.ones((1, input_dim[0] // 2,
                                          input_dim[1],
                                          input_dim[2]), dtype=torch.float32),
                              torch.zeros((1, input_dim[0] // 2,
                                           input_dim[1],
                                           input_dim[2]), dtype=torch.float32)], dim=1)
        else:
            mask = torch.cat([torch.zeros((1, input_dim[0] // 2,
                                           input_dim[1],
                                           input_dim[2]), dtype=torch.float32),
                              torch.ones((1, input_dim[0] // 2,
                                          input_dim[1],
                                          input_dim[2]), dtype=torch.float32)], dim=1)

        return mask

    def forward(self, x, reverse=False):
        self.mask = self.mask.to(x.device)
        x_masked = x * self.mask

        s = self.s_net(x_masked)
        t = self.t_net(x_masked)
        #print(s)

        s = s.tanh() * self.scale_scale + self.shift_scale
        #s = s.tanh() * self.scale_scale

        # umgedrehte mask
        anti_mask = 1 - self.mask

        s = s * anti_mask
        t = t * anti_mask

        if reverse:
            x = (x - t) * torch.exp(-s)
            return x

        else:
            x = x * s.exp() + t
            # f(x) = x*e^s + t => log(df/dx) = s
            return x, s