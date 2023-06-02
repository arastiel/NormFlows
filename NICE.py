import torch
import torch.nn as nn

from utillayers import Preprocess, Preprocess2, RescaleLayer
from couplinglayers import CouplingLayer

class NICE(nn.Module):
    """
    NICE model
    Inputs:
        input_dim - (CxHxW)
    """
    def __init__(self, input_dim):
        super(NICE, self).__init__()
        self.preprocess = Preprocess2()
        planes = 1000

        self.transforms = nn.ModuleList([
            CouplingLayer("checkerboard", input_dim, False, c_in=input_dim[0], c_hidden=planes),
            RescaleLayer(input_dim),
            CouplingLayer("checkerboard", input_dim, True, c_in=input_dim[0], c_hidden=planes),
            RescaleLayer(input_dim),
            CouplingLayer("checkerboard", input_dim, True, c_in=input_dim[0], c_hidden=planes),
            RescaleLayer(input_dim),
            CouplingLayer("checkerboard", input_dim, False, c_in=input_dim[0], c_hidden=planes),
            RescaleLayer(input_dim),
            #RescaleLayer(input_dim)
        ])

    def forward(self, x):
        z, log_det_jacobian_z = x, torch.zeros_like(x)
        #z, log_det_jacobian = self.preprocess(z)
        #log_det_jacobian_z += log_det_jacobian
        for transform in self.transforms:
            z, log_det_jacobian = transform(z)
            log_det_jacobian_z += log_det_jacobian
        return z, log_det_jacobian_z

    def inverse(self, z):
        x = z
        for transform in self.transforms[::-1]:
            x = transform(x, reverse=True)
        #x = self.preprocess(x, reverse=True)
        return x
