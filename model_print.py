from torchinfo import summary
from model.All_a_UNet import all_attention_UNet
import timm
import torch

model = all_attention_UNet(decode_channels=64,pretrained=False,num_classes=6)
summary(model,(2,3,512,512))

