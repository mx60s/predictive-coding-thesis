import torch
from torch import nn, optim
import torch.nn.functional as F

device = 'cuda'

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
        #print('start enc', torch.cuda.memory_allocated(device))
        x = self.bn1(torch.relu(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = F.adaptive_avg_pool2d(x, 1) # this sets it to [channel, 1, 1]
        x = x.view(x.size(0), -1)
        #print('end enc', torch.cuda.memory_allocated(device))
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
        #print('start dec', torch.cuda.memory_allocated(device))
        x = self.linear(x)
        x = x.view(x.size(0), 1024, 1, 1)
        x = F.interpolate(x, scale_factor=8)
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)
        x = torch.sigmoid(self.conv1(x))
        x = x.view(x.size(0), 3, 128, 128)
        #print('end dec', torch.cuda.memory_allocated(device))
        return x

# TODO is causal???
class MySelfAttention(nn.Module):
    def __init__(self, embed_dim=128, heads=8):
        super().__init__()
        self.conv1 = nn.Conv1d(7, 7, kernel_size=3, stride=4) # contract from [7, 512] to [7, 128]
        self.attn = nn.MultiheadAttention(embed_dim, heads, batch_first=True)
        # specifically in the paper they say that this is a conv but...
        # no matter what, the prediction is just a 1D vector
        # so does it make most sense to do a linear ff 

    def forward(self, x):
        #print('start attn', torch.cuda.memory_allocated(device))
        x = self.conv1(x)
        out_attn, _ = self.attn(x, x, x, need_weights=False)
        #print('out attn', torch.cuda.memory_allocated(device))
        return out_attn[:, -1, :]

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
        self.attn = MySelfAttention()
        self.decoder = ResNet18Dec()

    def forward(self, x):
        torch.cuda.empty_cache()
        encoded_frames = [self.encoder(frame.squeeze(1)) for frame in x.split(1, dim=1)]
        encoded_frames = torch.stack(encoded_frames, dim=1).squeeze(2)
        
        z = self.attn(encoded_frames)
        
        pred = self.decoder(z)
        return pred

# TODO: should this also be tasked to predict the head direction of the sample?
class LocationPredictor(nn.Module):
    """
    A simple feedforward network which predicts the position of the agent from a set of latent variables
    """
    def __init__(self, latent_model: PredictiveCoder , input_dim=128, hidden_dim=200):
        super().__init__()
        self.encoder = latent_model.encoder
        self.attn = latent_model.attn
        
        self.layer1 = nn.Linear(input_dim, 200)
        self.layer2 = nn.Linear(200, 2)
        
    def forward(self, x):
        with torch.no_grad():
            encoded_frames = [self.encoder(frame.squeeze(1)) for frame in x.split(1, dim=1)]
            encoded_frames = torch.stack(encoded_frames, dim=1).squeeze(2)
            z = self.attn(encoded_frames)
        
        out = F.relu(self.layer1(z))
        out = self.layer2(out)
        
        return out