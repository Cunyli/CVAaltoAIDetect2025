import torch
import torch.nn as nn
import torchvision

class ResNet34Segmentation(nn.Module):
    def __init__(self, input_channels=4, pretrained=True):
        super(ResNet34Segmentation, self).__init__()
        # 加载预训练的 resnet34 模型
        self.encoder = torchvision.models.resnet34(pretrained=pretrained)
        # 修改第一层卷积以适应4通道输入
        if input_channels != 3:
            original_conv = self.encoder.conv1
            self.encoder.conv1 = nn.Conv2d(input_channels,
                                           original_conv.out_channels,
                                           kernel_size=original_conv.kernel_size,
                                           stride=original_conv.stride,
                                           padding=original_conv.padding,
                                           bias=original_conv.bias is not None)
            # 将预训练权重复制到前3通道，第四通道可以随机初始化
            with torch.no_grad():
                self.encoder.conv1.weight[:, :3, :, :] = original_conv.weight
        # 保留 ResNet34 的前几层（去掉 avgpool 和 fc 层）
        self.encoder_layers = nn.Sequential(
            self.encoder.conv1,    # [B, 64, 128, 128] (256/2)
            self.encoder.bn1,
            self.encoder.relu,
            self.encoder.maxpool,  # [B, 64, 64, 64]  (256/4)
            self.encoder.layer1,   # [B, 64, 64, 64]
            self.encoder.layer2,   # [B, 128, 32, 32]
            self.encoder.layer3,   # [B, 256, 16, 16]
            self.encoder.layer4    # [B, 512, 8, 8]
        )
        # 构建反卷积 decoder，从 8x8 上采样到 256x256
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),  # 8x8 -> 16x16
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # 16x16 -> 32x32
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),   # 32x32 -> 64x64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),    # 64x64 -> 128x128
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),    # 128x128 -> 256x256
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1)  # 输出 1 通道
        )
    
    def forward(self, x):
        features = self.encoder_layers(x)
        out = self.decoder(features)
        return out

# 损失函数：使用带 logits 的二分类交叉熵损失
criterion = nn.BCEWithLogitsLoss()

# 示例：正向传播与损失计算
if __name__ == '__main__':
    # 假设 batch_size 为 2，输入形状 [2, 4, 256, 256]
    input_tensor = torch.randn(2, 4, 256, 256)
    # 创建模型
    model = ResNet34Segmentation(input_channels=4, pretrained=True)
    # 得到预测 soft mask，形状 [2, 1, 256, 256]
    output = model(input_tensor)
    print("输出形状:", output.shape)
    
    # 构造模拟的真实掩码 (0/1)，形状 [2, 1, 256, 256]
    true_mask = torch.randint(0, 2, (2, 1, 256, 256)).float()
    
    # 计算损失
    loss = criterion(output, true_mask)
    print("损失:", loss.item())
