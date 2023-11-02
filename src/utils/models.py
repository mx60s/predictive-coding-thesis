import torch
from torch import nn, optim
import torch.nn.functional as F

# TODO need to contend with head direction happening 

'''
To perform predictive coding, we construct an encoderdecoder convolutional neural network with a ResNet18 architecture40 for the encoder and a corresponding ResNet-18 architecture with transposed convolutions in the decoder (Figure 1(b)). The encoder-decoder architecture uses the U-Net architecture41 to pass the encoded latent units into the decoder. Multi-headed attention processes the sequence of encoded latent units to encode the history of past visual observations. The multi-headed attention has ℎ = 8 heads. For the encoded latent units with dimension 𝐷 = 𝐶 × 𝐻 × 𝑊, the dimension 𝑑 of a single head is 𝑑 = 𝐶 × 𝐻 × 𝑊/ℎ. The predictive coder approximates predictive coding by minimizing the mean-squared error between the Pre-print |

actual observation and its predicted observation. The predictive coder trains on 82, 630 samples for 200 epochs with gradient descent optimization with Nesterov momentum43, a weight decay of 5 × 10−6, and a learning rate of 10−1 adjusted by OneCycle learning rate scheduling44. The optimized predictive coder has to a mean-squared error between the predicted and actual images of 0.094 and a good visual fidelity (Figure 1(c)).
'''


class LocationPredictor(nn.Module):
    """
    A simple feedforward network which predicts the position of the agent from a set of latent variables
    """
    def __init__(self, input_dim=128, hidden_dim=200):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, 200)
        self.layer2 = nn.Linear(200, 2)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


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
        out = self.bn2(torch.relu(self.conv2(x)))
        out = self.bn1(self.conv1(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class ResNet18Enc(nn.Module):
    def __init__(self, num_Blocks=[2,2,2,2], nc=3):
        super().__init__()
        self.in_planes = 128
        # also the paper doesn't mention this first convolution
        # honestly not going to mess with this now because it doesn't seem like the most important thing
        self.conv1 = nn.Conv2d(nc, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.layer1 = self._make_layer(BasicBlockEnc, 128, num_Blocks[0], stride=1)
        self.layer2 = self._make_layer(BasicBlockEnc, 256, num_Blocks[1], stride=2)
        self.layer3 = self._make_layer(BasicBlockEnc, 512, num_Blocks[2], stride=2)
        self.layer4 = self._make_layer(BasicBlockEnc, 512, num_Blocks[3], stride=1)

    def _make_layer(self, BasicBlockEnc, planes, num_Blocks, stride):
        strides = [stride] + [1]*(num_Blocks-1)
        layers = []
        for stride in strides:
            layers += [BasicBlockEnc(self.in_planes, stride)]
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.bn1(torch.relu(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = F.adaptive_avg_pool2d(x, 1) # this sets it to [channel, 1, 1]
        x = x.view(x.size(0), -1)
        return x 


class ResNet18Dec(nn.Module):
    def __init__(self, z_dim=128, num_Blocks=[2,2,2,2], nc=3):
        super().__init__()
        self.in_planes = 1024

        # this is set so high so you can have the first stride=2, so it expands the H,W dims more
        # accurate to the referenced ResNet
        self.linear = nn.Linear(z_dim, 1024)

        self.layer4 = self._make_layer(BasicBlockDec, 512, num_Blocks[3], stride=2)
        self.layer3 = self._make_layer(BasicBlockDec, 256, num_Blocks[2], stride=2)
        self.layer2 = self._make_layer(BasicBlockDec, 128, num_Blocks[1], stride=2)
        self.layer1 = self._make_layer(BasicBlockDec, 128, num_Blocks[0], stride=1)
        self.conv1 = ResizeConv2d(128, nc, kernel_size=3, scale_factor=2)

    def _make_layer(self, BasicBlockDec, planes, num_Blocks, stride):
        strides = [stride] + [1]*(num_Blocks-1)
        layers = []
        for stride in reversed(strides):
            layers += [BasicBlockDec(self.in_planes, stride)]
        self.in_planes = planes
        return nn.Sequential(*layers)

    # feels bad to interpolate up so much. Should I try to just run this on 64x64 imgs instead?
    def forward(self, x):
        x = self.linear(x)
        x = x.view(x.size(0), 1024, 1, 1)
        x = F.interpolate(x, scale_factor=8)
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)
        x = torch.sigmoid(self.conv1(x))
        x = x.view(x.size(0), 3, 128, 128)
        return x

# TODO is causal???
class MySelfAttention(nn.Module):
    def __init__(self, latent_dim, embed_dim=128, heads=8):
        super().__init__()
        self.conv1 = nn.Conv1d(7, 7, kernel_size=3, stride=4) # contract from [7, 512] to [7, 128]
        self.attn = nn.MultiheadAttention(embed_dim, heads, batch_first=True)
        # specifically in the paper they say that this is a conv but...
        # no matter what, the prediction is just a 1D vector
        # so does it make most sense to do a linear ff 

    def forward(self, x):
        x = self.conv1(x)
        out_attn, _ = self.attn(x, x, x, need_weights=False)
        
        return out_attn[:, -1, :]

# the other issue is that a U Net is mentioned but not described. I'm gonna leave
# it out for now but it might make a big difference
# also they say UNEt just to pass into decoder so maybe there's a specific way to do that
class BasicAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ResNet18Enc()
        self.decoder = ResNet18Dec()

    def forward(self, x):
        # I'm changing this to just be an AE so it matches the architecture better
        # still wondering if this is a U Net as well
        z = self.encoder(x)
        x = self.decoder(z)
        return x#, z


class PredictiveCoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ResNet18Enc()
        self.attn = MySelfAttention(512)
        self.decoder = ResNet18Dec()

    def forward(self, x):
        # still wondering if this is a U Net as well
        encoded_frames = [self.encoder(frame.squeeze(1)) for frame in x.split(1, dim=1)]
        encoded_frames = torch.stack(encoded_frames, dim=1).squeeze(2)
        
        z = self.attn(encoded_frames)
        
        pred = self.decoder(z)
        return pred
