import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F

class ReNet(nn.Module):

    def __init__(self, n_input, n_units, usegpu=True):
        super(ReNet, self).__init__()

        self.rnn_hor = nn.GRU(n_input * 2 * 2, n_units, num_layers=1, batch_first=True, bidirectional=True)
        self.rnn_ver = nn.GRU(n_units * 2, n_units, num_layers=1, batch_first=True, bidirectional=True)

    def tile(self, x):

        n_height_padding = x.size(2) % 2
        n_width_padding = x.size(3) % 2

        x = F.pad(x, (0, n_width_padding, 0, n_height_padding))

        b, n_filters, n_height, n_width = x.size()

        assert n_height % 2 == 0
        assert n_width & 2 == 0

        new_height = n_height / 2
        new_width = n_width / 2

        x = x.view(b, n_filters, new_height, 2, new_width, 2)
        x = x.permute(0, 2, 4, 1, 3, 5)
        x = x.contiguous()
        x = x.view(b, new_height, new_width, 2 * 2 * n_filters)
        x = x.permute(0, 3, 1, 2)
        x = x.contiguous()

        return x

    def rnn_forward(self, x, hor_or_ver):

        assert hor_or_ver in ['hor', 'ver']

        b, n_height, n_width, n_filters = x.size()

        x = x.view(b * n_height, n_width, n_filters)
        if hor_or_ver == 'hor':
            x, _ = self.rnn_hor(x)
        else:
            x, _ = self.rnn_ver(x)
        x = x.contiguous()
        x = x.view(b, n_height, n_width, -1)

        return x

    def forward(self, x):

                                       #b, nf, h, w
        x = self.tile(x)               #b, nf, h, w
        x = x.permute(0, 2, 3, 1)      #b, h, w, nf
        x = x.contiguous()
        x = self.rnn_forward(x, 'hor') #b, h, w, nf
        x = x.permute(0, 2, 1, 3)      #b, w, h, nf
        x = x.contiguous()
        x = self.rnn_forward(x, 'ver') #b, w, h, nf
        x = x.permute(0, 2, 1, 3)      #b, h, w, nf
        x = x.contiguous()
        x = x.permute(0, 3, 1, 2)      #b, nf, h, w
        x = x.contiguous()

        return x

class CNN(nn.Module):

    def __init__(self, usegpu=True):
        super(CNN, self).__init__()

        self.model = models.__dict__['resnet50'](pretrained=True)
        self.model = nn.Sequential(*list(self.model.children())[:-4])

    def forward(self, x):

        b, n_channel, n_height, n_width = x.size()
        x = self.model(x)

        return x

class Architecture(nn.Module):

    def __init__(self, n_classes, usegpu=True):
        super(Architecture, self).__init__()

        self.n_classes = n_classes

        self.cnn = CNN(usegpu=usegpu)
        self.renet1 = ReNet(512, 256, usegpu=usegpu)
        self.renet2 = ReNet(256 * 2, 256, usegpu=usegpu)
        self.upsampling1 = nn.ConvTranspose2d(256 * 2, 256, kernel_size=(10, 18), stride=(3, 4))
        self.upsampling2 = nn.ConvTranspose2d(256, 256, kernel_size=(10, 13), stride=(3, 3))
        self.upsampling3 = nn.ConvTranspose2d(256, 128, kernel_size=(25, 10), stride=(3, 2))
        self.upsampling4 = nn.ConvTranspose2d(128, self.n_classes, kernel_size=(20, 17), stride=(1, 1))

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()

    def forward(self, x):
        x = self.cnn(x)
        x = self.renet1(x)
        x = self.renet2(x)
        x = self.relu1(self.upsampling1(x))
        x = self.relu2(self.upsampling2(x))
        x = self.relu3(self.upsampling3(x))
        x = self.upsampling4(x)
        return x
