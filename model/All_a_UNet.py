import timm
import torch
import torch.nn as nn
from timm.layers.norm import GroupNorm
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, channel, r=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel//r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel//r, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c , _, _ = x.size()
        # Squeeze
        y = self.avg_pool(x).view(b, c)
        # Excitation
        y = self.fc(y).view(b, c, 1, 1)
        # Fscale
        y = torch.mul(x, y)
        return y
    
class SpaceBlock(nn.Module):
    def __init__(self, channel):
        super(SpaceBlock, self).__init__()
        
        self.space = nn.Sequential(
            GroupNorm(num_channels=channel),
            Conv(in_channels=channel,out_channels=channel), 
            
        )
        self.sigmoidrelu =  nn.Sequential(
            nn.Sigmoid(),
            nn.ReLU6(), 
            
        )
        

    def forward(self, x,res):
        #x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.space(self.sigmoidrelu(self.space(x)+self.space(res)))*x
        
        return x 
    
class SpaceChannel(nn.Module):
    def __init__(self, in_channel,channel):
        super(SpaceChannel, self).__init__()
        self.CNN = nn.Conv2d(in_channels=in_channel,out_channels=channel,kernel_size=1)
        self.space = SpaceBlock(channel=channel)
        self.channel = SEBlock(channel=channel)
 
    def forward(self, x,res):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        res = self.CNN(res)
        space = self.space(x,res)
        x = self.channel(x)+self.channel(space)
        
        return x 

class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )

class head_GN(nn.Module):
    def __init__(self, in_channel=3,out_channel=64):
        super(head_GN, self).__init__()
        
        self.GN = nn.Sequential(
            Conv(in_channels=in_channel,out_channels=out_channel),
            GroupNorm(num_channels=out_channel),
            Conv(in_channels=out_channel,out_channels=out_channel),
            GroupNorm(num_channels=out_channel),
        )

    def forward(self, x):

        x = self.GN(x)

        return x
        
    
class all_attention_UNet(nn.Module):
    def __init__(self,
                 decode_channels=64,
                 backbone_name='resnest50d',
                 pretrained=True,
                 num_classes=6
                 ):
        super(all_attention_UNet,self).__init__()

        self.backbone = timm.create_model(backbone_name, features_only=True, output_stride=32,
                                          out_indices=(0, 1, 2, 3, 4), pretrained=pretrained)
        encoder_channels = self.backbone.feature_info.channels()
        self.head = head_GN(in_channel=3,out_channel=decode_channels)
        self.pre_conv = nn.Conv2d(in_channels=encoder_channels[-1],out_channels=decode_channels,kernel_size=1)
        self.b4 = SpaceChannel(encoder_channels[-2],channel=decode_channels)
        self.b3 = SpaceChannel(encoder_channels[-3],channel=decode_channels)
        self.b2 = SpaceChannel(encoder_channels[-4],channel=decode_channels)
        self.b1 = SpaceChannel(encoder_channels[-5],channel=decode_channels)
        
        self.b_head = SpaceChannel(decode_channels,channel=decode_channels)
        self.classes = nn.Conv2d(in_channels=decode_channels,out_channels=num_classes,kernel_size=1)

    def forward(self, x):
        head = self.head(x)
        res0,res1,res2,res3,res4  = self.backbone(x)
        
        x = self.b4(self.pre_conv(res4),res3)
        x = self.b3(x,res2)
        x = self.b2(x,res1)
        x = self.b1(x,res0)
        
        x = self.b_head(x,head)
        x = self.classes(x)
        return x