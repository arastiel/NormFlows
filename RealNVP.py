import torch
import torch.nn as nn

from utillayers import Preprocess, LinearBatchNorm
from couplinglayers import CouplingLayer2

class RealNVP(nn.Module):
    """
    RealNVP model
    Inputs:
        input_dim - (CxHxW)
    """
    def __init__(self, input_dim):
        super(RealNVP, self).__init__()
        # (B, 1, 28, 28)
        self.preprocess = Preprocess()
        planes = 64

        self.transforms_checkered_1 = nn.ModuleList([
            CouplingLayer2("checkerboard", input_dim, False, c_in=input_dim[0], c_hidden=planes),
            LinearBatchNorm(input_dim),
            CouplingLayer2("checkerboard", input_dim, True, c_in=input_dim[0], c_hidden=planes),
            LinearBatchNorm(input_dim),
            CouplingLayer2("checkerboard", input_dim, False, c_in=input_dim[0], c_hidden=planes),
            LinearBatchNorm(input_dim),
        ])

        input_dim = (input_dim[0] * 4, input_dim[1] // 2, input_dim[2] // 2)
        # (B, 4, 14, 14)
        self.transforms_channel_1 = nn.ModuleList([
            CouplingLayer2("channel", input_dim, True, c_in=input_dim[0], c_hidden=planes),
            LinearBatchNorm(input_dim),
            CouplingLayer2("channel", input_dim, False, c_in=input_dim[0], c_hidden=planes),
            LinearBatchNorm(input_dim),
            CouplingLayer2("channel", input_dim, True, c_in=input_dim[0], c_hidden=planes),
            LinearBatchNorm(input_dim),
        ])

        input_dim = (input_dim[0] // 2, input_dim[1], input_dim[2])
        planes = planes * 2
        # (B, 2, 14, 14)
        self.transforms_checkered_2 = nn.ModuleList([
            CouplingLayer2("checkerboard", input_dim, False, c_in=input_dim[0], c_hidden=planes),
            LinearBatchNorm(input_dim),
            CouplingLayer2("checkerboard", input_dim, True, c_in=input_dim[0], c_hidden=planes),
            LinearBatchNorm(input_dim),
            CouplingLayer2("checkerboard", input_dim, False, c_in=input_dim[0], c_hidden=planes),
            LinearBatchNorm(input_dim),
        ])

        input_dim = (input_dim[0] * 4, input_dim[1] // 2, input_dim[2] // 2)
        # (B, 8, 7, 7)
        self.transforms_channel_2 = nn.ModuleList([
            CouplingLayer2("channel", input_dim, True, c_in=input_dim[0], c_hidden=planes),
            LinearBatchNorm(input_dim),
            CouplingLayer2("channel", input_dim, False, c_in=input_dim[0], c_hidden=planes),
            LinearBatchNorm(input_dim),
            CouplingLayer2("channel", input_dim, True, c_in=input_dim[0], c_hidden=planes),
            LinearBatchNorm(input_dim),
        ])

        input_dim = (input_dim[0] // 2, input_dim[1], input_dim[2])
        planes = planes * 4
        # (B, 4, 7, 7)
        self.transforms_checkered_3 = nn.ModuleList([
            CouplingLayer2("checkerboard", input_dim, False, c_in=input_dim[0], c_hidden=planes),
            LinearBatchNorm(input_dim),
            CouplingLayer2("checkerboard", input_dim, True, c_in=input_dim[0], c_hidden=planes),
            LinearBatchNorm(input_dim),
            CouplingLayer2("checkerboard", input_dim, False, c_in=input_dim[0], c_hidden=planes),
            LinearBatchNorm(input_dim),
        ])

    def squeeze(self, x):
        '''converts a (batch_size,1,4,4) tensor into a (batch_size,4,2,2) tensor'''
        batch_size, num_channels, height, width = x.size()
        x = x.reshape(batch_size, num_channels, height // 2, 2, width // 2, 2)
        x = x.permute(0, 1, 3, 5, 2, 4)
        x = x.reshape(batch_size, num_channels * 4, height // 2, width // 2)
        return x

    def unsqueeze(self, x):
        '''converts a (batch_size,4,2,2) tensor into a (batch_size,1,4,4) tensor'''
        batch_size, num_channels, height, width = x.size()
        x = x.reshape(batch_size, num_channels // 4, 2, 2, height, width)
        x = x.permute(0, 1, 4, 2, 5, 3)
        x = x.reshape(batch_size, num_channels // 4, height * 2, width * 2)
        return x

    def half_input(self, x):
        factor_channels = x.shape[1] // 2
        x_flat = torch.flatten(x[:, factor_channels:, :, :], 1)
        firhalf = x[:, :factor_channels, :, :]
        sechalf = x[:, factor_channels:, :, :]
        return firhalf, sechalf

    def unhalf_input(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        return x

    def forward(self, x):
        z, log_det_jacobian_z = x, torch.zeros_like(x)

        #z, log_det_jacobian = self.preprocess(z)
        #log_det_jacobian_z += log_det_jacobian

        for transform in self.transforms_checkered_1:
            z, log_det_jacobian = transform(z)
            log_det_jacobian_z += log_det_jacobian

        # (B, 1, 28, 28) -> (B, 4, 14, 14)
        z = self.squeeze(z)  # anpassen für channelwise

        log_det_jacobian_z = self.squeeze(log_det_jacobian_z)

        for transform in self.transforms_channel_1:
            z, log_det_jacobian = transform(z)
            log_det_jacobian_z += log_det_jacobian

        # (b, 4, 14, 14) -> (b, 2, 14, 14)
        z1, z2 = self.half_input(z)
        log_det_jacobian_z1, log_det_jacobian_z2 = self.half_input(log_det_jacobian_z)

        for transform in self.transforms_checkered_2:
            z1, log_det_jacobian = transform(z1)
            log_det_jacobian_z1 += log_det_jacobian

        # (b,2,14,14) -> (b, 8, 7, 7)
        z1 = self.squeeze(z1)
        log_det_jacobian_z1 = self.squeeze(log_det_jacobian_z1)

        for transform in self.transforms_channel_2:
            z1, log_det_jacobian = transform(z1)
            log_det_jacobian_z1 += log_det_jacobian

        # (b, 8, 7, 7) -> (b, 4, 7, 7)
        z3, z4 = self.half_input(z1)
        log_det_jacobian_z3, log_det_jacobian_z4 = self.half_input(log_det_jacobian_z1)

        for transform in self.transforms_checkered_3:
            z3, log_det_jacobian = transform(z3)
            log_det_jacobian_z3 += log_det_jacobian

        # (b, 4, 7, 7) -> (b, 8, 7, 7)
        z1 = self.unhalf_input(z3, z4)
        log_det_jacobian_z1 = self.unhalf_input(log_det_jacobian_z3, log_det_jacobian_z4)

        # (b, 8, 7, 7) -> (b, 2, 14, 14)
        z1 = self.unsqueeze(z1)
        log_det_jacobian_z1 = self.unsqueeze(log_det_jacobian_z1)

        # (b, 2, 14, 14) -> (b, 4, 14, 14)
        z = self.unhalf_input(z1, z2)
        log_det_jacobian_z = self.unhalf_input(log_det_jacobian_z1, log_det_jacobian_z2)

        # (b, 4, 14, 14) -> (b, 1, 28, 28)
        z = self.unsqueeze(z)
        log_det_jacobian_z = self.unsqueeze(log_det_jacobian_z)

        return z, log_det_jacobian_z

    def inverse(self, z):
        x = z

        # (b,1,28,28) -> (b,4,14,14)
        x = self.squeeze(x)

        # (b,4,14,14) -> (b,2,14,14)
        x1, x2 = self.half_input(x)

        # (b, 2, 14, 14) -> (b, 8, 7, 7)
        x1 = self.squeeze(x1)

        # (b, 8, 7, 7) -> (b, 4, 7, 7)
        x3, x4 = self.half_input(x1)

        for transform in self.transforms_checkered_3[::-1]:
            x3 = transform(x3, reverse=True)

        # (b, 4, 7, 7) -> (b, 8, 7, 7)
        x1 = self.unhalf_input(x3, x4)

        # print(x1.size())
        for transform in self.transforms_channel_2[::-1]:
            x1 = transform(x1, reverse=True)

        # (b, 8, 7, 7) -> (b, 2, 14, 14)
        x1 = self.unsqueeze(x1)

        for transform in self.transforms_checkered_2[::-1]:
            x1 = transform(x1, reverse=True)

        # (b, 2, 14, 14) -> (b, 4, 14, 14)
        x = self.unhalf_input(x1, x2)

        for transform in self.transforms_channel_1[::-1]:
            x = transform(x, reverse=True)

        # (b, 4, 14, 14) -> (b, 1, 28, 28)
        x = self.unsqueeze(x)

        for transform in self.transforms_checkered_1[::-1]:
            x = transform(x, reverse=True)

        #x = self.preprocess(x, reverse=True)

        return x