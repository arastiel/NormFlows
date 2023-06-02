import torch.nn as nn

from utillayers import WeightNormConv2d

class m_net(nn.Module):
    """
    m_net for NN in NICE

    transformation batch norm
    Inputs:
        c_in - number input features
        c_hidden - number hidden features
    """
    def __init__(self, c_in, c_hidden=32, c_out=-1, num_layers=5):
        super().__init__()
        c_out = c_out if c_out > 0 else c_in
        layers = []
        layers.append(nn.Linear(c_in, c_hidden))
        layers.append(nn.ReLU())
        for i in range(num_layers):
            layers.append(nn.Linear(c_hidden, c_hidden))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(c_hidden, c_in))
        self.network = nn.Sequential(*layers)
        #print(self.network)

    def forward(self, x):
        og_shape = x.shape
        x = x.view(og_shape[0], -1)
        #print(x.shape)
        z = self.network(x)
        z = z.view(og_shape)
        return z

class EasyNetBlock(nn.Module):
    """
    block for building NN in RealNVP
    Inputs:
        c_in - number input features
        c_hidden - number hidden features
    """
    def __init__(self, c_in, c_hidden):
        super().__init__()
        self.net = nn.Sequential(
            WeightNormConv2d(in_channel=c_in, out_channel=c_hidden, kernel_size=3, stride=1, padding=1),
            # nn.Conv2d(c_in, c_hidden, kernel_size=1, stride=1, padding=0),
            nn.InstanceNorm2d(c_hidden),
            # LayerNorm(c_hidden),
            nn.ReLU(),
            # WeightNormConv2d(in_channel=c_in, out_channel=c_hidden, kernel_size=1, stride=1, padding=0),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.net(x)
        x = x + y
        x = self.relu(x)
        return x


class EasyNet(nn.Module):
    """
    s/t_net for NN in RealNVP
    Inputs:
        c_in - number input features
        c_hidden - number hidden features
    """
    def __init__(self, c_in, c_hidden=32, c_out=-1, num_layers=2):
        super().__init__()
        c_out = c_out if c_out > 0 else c_in
        layers = []
        layers.append(nn.Conv2d(c_in, c_hidden, kernel_size=3, padding=1))
        for i in range(num_layers):
            layers.append(EasyNetBlock(c_hidden, c_hidden))
            # layers.append(nn.BatchNorm2d(c_hidden))
        layers.append(nn.Conv2d(c_hidden, c_out, kernel_size=3, padding=1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)