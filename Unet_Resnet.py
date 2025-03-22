import torch
import torch.nn as nn
from torchvision import transforms, models

# ---------------------------
# 定义 ResNet34-U-Net 模型
# ---------------------------
class ResNetUNet(nn.Module):
    def __init__(self, out_channels=1):
        super(ResNetUNet, self).__init__()
        # 使用预训练的 ResNet34 作为 encoder
        self.base_model = models.resnet34(pretrained=True)
        base_layers = list(self.base_model.children())
        # layer0：conv1 + bn1 + relu，输入尺寸：256x256 -> 128x128
        self.layer0 = nn.Sequential(base_layers[0], base_layers[1], base_layers[2])
        # maxpool: 128x128 -> 64x64
        self.maxpool = base_layers[3]
        # layer1：64通道，输出尺寸保持 64x64
        self.layer1 = base_layers[4]
        # layer2：128通道，64x64 -> 32x32
        self.layer2 = base_layers[5]
        # layer3：256通道，32x32 -> 16x16
        self.layer3 = base_layers[6]
        # layer4：512通道，16x16 -> 8x8
        self.layer4 = base_layers[7]
        
        # ---------------------------
        # 解码器部分
        # ---------------------------
        # 从 layer4 到 layer3：8x8 -> 16x16，拼接后通道 512（256+256），卷积后输出256
        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv4 = self.conv_block(512, 256)
        # 从 layer3 到 layer2：16x16 -> 32x32，拼接后通道 256（128+128），卷积后输出128
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = self.conv_block(256, 128)
        # 从 layer2 到 layer1：32x32 -> 64x64，拼接后通道 128（64+64），卷积后输出64
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv2 = self.conv_block(128, 64)
        # 从 layer1 到 layer0：64x64 -> 128x128，拼接后通道 128（64+64），卷积后输出64
        self.upconv1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.conv1 = self.conv_block(128, 64)
        # 最后上采样至256x256
        self.upconv0 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def conv_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        return block

    def forward(self, x):
        # Encoder：ResNet34
        x0 = self.layer0(x)         # [B, 64, 128, 128]
        x1 = self.maxpool(x0)       # [B, 64, 64, 64]
        x2 = self.layer1(x1)        # [B, 64, 64, 64]
        x3 = self.layer2(x2)        # [B, 128, 32, 32]
        x4 = self.layer3(x3)        # [B, 256, 16, 16]
        x5 = self.layer4(x4)        # [B, 512, 8, 8]
        
        # Decoder
        d4 = self.upconv4(x5)       # [B, 256, 16, 16]
        d4 = torch.cat([d4, x4], dim=1)  # 拼接: [B, 512, 16, 16]
        d4 = self.conv4(d4)         # [B, 256, 16, 16]
        
        d3 = self.upconv3(d4)       # [B, 128, 32, 32]
        d3 = torch.cat([d3, x3], dim=1)  # 拼接: [B, 256, 32, 32]
        d3 = self.conv3(d3)         # [B, 128, 32, 32]
        
        d2 = self.upconv2(d3)       # [B, 64, 64, 64]
        d2 = torch.cat([d2, x2], dim=1)  # 拼接: [B, 128, 64, 64]
        d2 = self.conv2(d2)         # [B, 64, 64, 64]
        
        d1 = self.upconv1(d2)       # [B, 64, 128, 128]
        d1 = torch.cat([d1, x0], dim=1)  # 拼接: [B, 128, 128, 128]
        d1 = self.conv1(d1)         # [B, 64, 128, 128]
        
        d0 = self.upconv0(d1)       # [B, 64, 256, 256]
        out = self.final_conv(d0)   # [B, out_channels, 256, 256]
        return out