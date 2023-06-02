import torch
from torch.distributions.normal import Normal
from torch.distributions.laplace import Laplace
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.uniform import Uniform
from torch.distributions.transforms import SigmoidTransform
from torch.distributions.transforms import AffineTransform

"""
dataset - celebA/mnist
model - NICE/RealNVP/RealNVP2
filename - savefile model
lr - learning rate optimizer
epochs_to_learn - epochs to learn once started
num_ex - numbers of generated images
"""

'''
pretrained models - RealNVP_celebA3.pt.tar
                    RealNVP_mnist3.pt.tar
                    NICE_celebA3.pt.tar
                    NICE_mnist3.pt.tar
                    RealNVP2_test.pt.tar
                    
---------- IMPORTANT DATA USED OF PRETRAINED MODELS ----------
                           
NICE celebA:
lr = 0.0004
b = 32
target_dist = logistic distri

NICE mnist:
lr = 0.0004
b = 64
target_dist = logistic distri
'''

'''
RealNVP celebA:
lr = 0.0005
b = 32
target_dist = logistic distri

RealNVP mnist:
lr = 0.0004
b = 64
target_dist = logistic dist


RealNVP2 mnist:
lr = 0.0004
b = 64
target_dist = logistic
'''


#dataset = 'celebA'
dataset = 'mnist'

model = 'NICE'
#model = 'RealNVP'
#model = 'RealNVP2'


if dataset=='mnist':
    img_size = 28
    channel = 1
    batch_size = 64

elif dataset == 'celebA':
    img_size = 64
    channel = 3
    batch_size = 32


#filename = "NICE_celebA.pt.tar"
#savefolder = "NICEcelebA/"

filename = "NICE_mnist3.pt.tar"
#filename = "testy.pt.tar"  #TEST
savefolder = "ULTRATEST/"

#filename = "NICE_mnist2.pt.tar"        #REAL
#savefolder = "NICEmnist2/"

lr = 0.0004
epochs_to_learn = 500
num_ex = 36  # 9, 16, 25, 36,....

device = 'cuda' if torch.cuda.is_available() else 'cpu'


#target_dist = Normal(0., 1.)
#target_dist = Laplace(-5., 4.)
''' logistic distribution https://pytorch.org/docs/stable/distributions.html '''
base_distribution = Uniform(torch.tensor(0.).to(device), torch.tensor(1.).to(device))
transforms = [SigmoidTransform().inv, AffineTransform(loc=0., scale=1.)]
target_dist = TransformedDistribution(base_distribution, transforms)
