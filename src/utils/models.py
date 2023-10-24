import torch
from torch import nn, optim
import torch.nn.functional as F

# TODO make a autoencoder and then switch to building out the envs

'''
To perform predictive coding, we construct an encoderdecoder convolutional neural network with a ResNet18 architecture40 for the encoder and a corresponding ResNet-18 architecture with transposed convolutions in the decoder (Figure 1(b)). The encoder-decoder architecture uses the U-Net architecture41 to pass the encoded latent units into the decoder. Multi-headed attention processes the sequence of encoded latent units to encode the history of past visual observations. The multi-headed attention has ℎ = 8 heads. For the encoded latent units with dimension 𝐷 = 𝐶 × 𝐻 × 𝑊, the dimension 𝑑 of a single head is 𝑑 = 𝐶 × 𝐻 × 𝑊/ℎ. The predictive coder approximates predictive coding by minimizing the mean-squared error between the Pre-print |

actual observation and its predicted observation. The predictive coder trains on 82, 630 samples for 200 epochs with gradient descent optimization with Nesterov momentum43, a weight decay of 5 × 10−6, and a learning rate of 10−1 adjusted by OneCycle learning rate scheduling44. The optimized predictive coder has to a mean-squared error between the predicted and actual images of 0.094 and a good visual fidelity (Figure 1(c)).
'''

# arch from here: https://github.com/julianstastny/VAE-ResNet18-PyTorch/tree/master
# with transposed convolutions subbed out -- may need to change that later

class ResizeConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x


class BasicBlockEnc(nn.Module):
    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = in_planes*stride

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        if stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        # some modifications from source code to fit the paper's description
        out = self.bn1(torch.relu(self.conv1(x))) 
        out = self.bn2(torch.relu(self.conv2(out)))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class BasicBlockDec(nn.Module):
    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = int(in_planes/stride)

        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_planes)
        # self.bn1 could have been placed here, but that messes up the order of the layers when printing the class

        if stride == 1:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential()
        else:
            self.conv1 = ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential(
                ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        # keeping this as original for now to fix issues
        # did not help lol
        out = torch.relu(self.bn2(self.conv2(x)))
        out = self.bn1(self.conv1(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class ResNet18Enc(nn.Module):
    def __init__(self, num_Blocks=[2,2,2,2], z_dim=10, nc=3):
        super().__init__()
        self.in_planes = 32
        self.z_dim = z_dim
        # also the paper doesn't mention this first convolution
        self.conv1 = nn.Conv2d(nc, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        # TODO: I'm pretty sure there are 2x the layers here for each size
        # nvm that's specified with numBlocks
        self.layer1 = self._make_layer(BasicBlockEnc, 32, num_Blocks[0], stride=1)
        self.layer2 = self._make_layer(BasicBlockEnc, 64, num_Blocks[1], stride=2)
        self.layer3 = self._make_layer(BasicBlockEnc, 128, num_Blocks[2], stride=2)
        self.layer4 = self._make_layer(BasicBlockEnc, 256, num_Blocks[3], stride=2)
        # do you need this linear layer at the end? I'm not sure
        self.linear = nn.Linear(256, 2 * z_dim)

    def _make_layer(self, BasicBlockEnc, planes, num_Blocks, stride):
        strides = [stride] + [1]*(num_Blocks-1)
        layers = []
        for stride in strides:
            layers += [BasicBlockEnc(self.in_planes, stride)]
            # so the first layer will have 2 basicblockenc layers, the first with planes=32 the second 128
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        #print(x.shape)
        x = self.layer1(x)
        #print(x.shape)
        x = self.layer2(x)
        #print(x.shape)
        x = self.layer3(x)
        #print(x.shape)
        x = self.layer4(x)
        #print(x.shape)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        mu = x[:, :self.z_dim]
        logvar = x[:, self.z_dim:]
        #print('mu', mu.shape)
        #print('logvar', logvar.shape)
        return mu, logvar


class ResNet18Dec(nn.Module):
    def __init__(self, num_Blocks=[2,2,2,2], z_dim=10, nc=3):
        super().__init__()
        self.in_planes = 256

        self.linear = nn.Linear(z_dim, 256)

        # layer sizes are pretty different here to fit the paper, might be misunderstanding how this wroks
        self.layer4 = self._make_layer(BasicBlockDec, 128, num_Blocks[3], stride=2)
        self.layer3 = self._make_layer(BasicBlockDec, 64, num_Blocks[2], stride=2)
        self.layer2 = self._make_layer(BasicBlockDec, 32, num_Blocks[1], stride=2)
        self.layer1 = self._make_layer(BasicBlockDec, 32, num_Blocks[0], stride=1)
        self.conv1 = ResizeConv2d(32, nc, kernel_size=3, scale_factor=2)

    def _make_layer(self, BasicBlockDec, planes, num_Blocks, stride):
        strides = [stride] + [1]*(num_Blocks-1)
        layers = []
        for stride in reversed(strides):
            layers += [BasicBlockDec(self.in_planes, stride)]
        self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, z):
        x = self.linear(z)
        x = x.view(z.size(0), 256, 1, 1)
        #print('x.view decoder', x.shape)
        x = F.interpolate(x, scale_factor=2)
        #print('x interpolate', x.shape)
        x = self.layer4(x)
        #print(x.shape)
        x = self.layer3(x)
        #print(x.shape)
        x = self.layer2(x)
        #print(x.shape)
        x = self.layer1(x)
        #print(x.shape)
        x = torch.sigmoid(self.conv1(x))
        #print('x.conv resize', x.shape)
        x = x.view(x.size(0), 3, 32, 32)
        return x

# the other issue is that a U Net is mentioned but not described. I'm gonna leave
# it out for now but it might make a big difference
class BasicVAE(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.encoder = ResNet18Enc(z_dim=z_dim)
        self.decoder = ResNet18Dec(z_dim=z_dim)

    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        #print('z', z.shape)
        x = self.decoder(z)
        return x, z
    
    @staticmethod
    # why reparameterize?
    def reparameterize(mean, logvar):
        std = torch.exp(logvar / 2) # in log-space, squareroot is divide by two
        epsilon = torch.randn_like(std)
        return epsilon * std + mean

