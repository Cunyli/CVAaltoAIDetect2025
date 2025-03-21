import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# -------------------------------
# 数据集定义
# -------------------------------
class ImageRestorationDataset(Dataset):
    def __init__(self, original_dir, modified_dir, mask_dir, transform=None):
        """
        参数:
          original_dir: 原始图片目录路径 (RGB图像)
          modified_dir: 修改后图片目录路径 (RGB图像)
          mask_dir:     mask 图片目录路径 (二值单通道图像)
          transform:    可选的 transform，用于后续处理（例如随机裁剪、归一化等）
        """
        self.original_dir = original_dir
        self.modified_dir = modified_dir
        self.mask_dir = mask_dir
        self.transform = transform
        
        # 假设三个目录中图片名称完全一致（例如 image_0.png, image_1.png, ...）
        self.filenames = sorted(os.listdir(modified_dir))
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        filename = self.filenames[idx]
        
        # 构造对应的路径
        original_path = os.path.join(self.original_dir, filename)
        modified_path = os.path.join(self.modified_dir, filename)
        mask_path = os.path.join(self.mask_dir, filename)
        
        # 加载图片：原始图片和修改后图片为 RGB；mask 为单通道灰度
        original_img = Image.open(original_path).convert("RGB")
        modified_img = Image.open(modified_path).convert("RGB")
        mask_img = Image.open(mask_path).convert("L")
        
        # 转换为 tensor（归一化到 [0,1]）
        to_tensor = transforms.ToTensor()
        original_tensor = to_tensor(original_img)       # shape: (3, H, W)
        modified_tensor = to_tensor(modified_img)       # shape: (3, H, W)
        mask_tensor = to_tensor(mask_img)               # shape: (1, H, W)
        
        # 保证 mask 为二值化（阈值0.5），结果为 0 或 1
        mask_tensor = (mask_tensor > 0.5).float()
        
        # 输入：将修改后的图片和 mask 拼接，形成 4 通道输入
        input_tensor = torch.cat([modified_tensor, mask_tensor], dim=0)  # shape: (4, H, W)
        
        # 目标输出：原始图片扩展为4通道，第四通道为全0（与输入保持一致的结构）
        zeros_channel = torch.zeros(1, original_tensor.shape[1], original_tensor.shape[2])
        target_tensor = torch.cat([original_tensor, zeros_channel], dim=0)  # shape: (4, H, W)
        
        if self.transform:
            input_tensor = self.transform(input_tensor)
            target_tensor = self.transform(target_tensor)
            
        return input_tensor, target_tensor

# -------------------------------
# 模型定义：Encoder 和 Decoder
# -------------------------------
class Encoder(nn.Module):
    def __init__(self, in_channels=4, latent_dim=64):
        """
        Encoder 模块：输入为 4 通道（修改后的 RGB + mask），输出 latent 特征
        """
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            # 输出尺寸：256 -> 128
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # 输出尺寸：128 -> 64
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # 输出尺寸：64 -> 32
            nn.Conv2d(64, latent_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, out_channels=4, latent_dim=64):
        """
        Decoder 模块：输入 latent 特征，输出重构的 4 通道图像（RGB + mask）
        """
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            # 输出尺寸：32 -> 64
            nn.ConvTranspose2d(latent_dim, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            # 输出尺寸：64 -> 128
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            # 输出尺寸：128 -> 256
            nn.ConvTranspose2d(32, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # 输出归一化到 [0,1]
        )
        
    def forward(self, x):
        return self.decoder(x)

# -------------------------------
# 整体网络（encoder + decoder）
# -------------------------------
class RestorationCNN(nn.Module):
    def __init__(self, in_channels=4, out_channels=4, latent_dim=64):
        super(RestorationCNN, self).__init__()
        self.encoder = Encoder(in_channels=in_channels, latent_dim=latent_dim)
        self.decoder = Decoder(out_channels=out_channels, latent_dim=latent_dim)
    
    def forward(self, x):
        latent = self.encoder(x)
        out = self.decoder(latent)
        return out

# -------------------------------
# DataLoader 以及训练示例
# -------------------------------
def main():
    # 设置数据目录（请根据实际情况修改）
    original_dir = "data/original"    # 原始图片目录
    modified_dir = "data/modified"    # 修改后图片目录
    mask_dir = "data/mask"            # mask 图片目录

    # 创建数据集和 DataLoader
    dataset = ImageRestorationDataset(original_dir, modified_dir, mask_dir)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)
    
    # 初始化模型、损失函数和优化器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RestorationCNN(in_channels=4, out_channels=4, latent_dim=64).to(device)
    criterion = nn.MSELoss()  # 重构损失
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # 简单的训练循环示例
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device)    # 输入 shape: (B, 4, 256, 256)
            targets = targets.to(device)  # 目标 shape: (B, 4, 256, 256)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}")
        
        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Average Loss: {epoch_loss:.4f}")
    
    # 训练完成后，可以保存模型
    torch.save(model.state_dict(), "restoration_cnn.pth")
    print("模型已保存。")

if __name__ == "__main__":
    main()
