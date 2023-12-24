# all_attention_UNet
# 复现All Attention U-NET for Semantic Segmentation of Intracranial Hemorrhages In Head CT Images
Paper：[https://arxiv.org/abs/2312.10483](https://arxiv.org/abs/2312.10483)
![image](https://github.com/Jacky-Android/all_attention_UNet/assets/55181594/ad420880-893a-4630-8bea-61001115b606)

在头部CT扫描中，颅内出血是专家诊断不同类型的首要工具。然而，它们的类型在同一类型中具有不同的形状，在类型之间的形状、大小和位置上也很相似，容易引起混淆。为了解决这个问题，本文提出了一种全注意力U-Net。它在U-Net编码器侧使用通道关注来增强类别特定特征提取，并在U-Net解码器侧使用空间和通道关注进行更精确的形状提取和类型分类。模拟结果显示，与基线ResNet50 + U-Net相比，改进了31.8%，而且性能优于有限关注情况下的情况。
# channel attention is SE Attention
```python
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
```
# Space Attention
![image](https://github.com/Jacky-Android/all_attention_UNet/assets/55181594/4a5ca899-0cee-44bd-85b0-820999923284)
```python
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
```
