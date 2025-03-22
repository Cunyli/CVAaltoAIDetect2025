import os
import argparse
import time
from pathlib import Path
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, models
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import wandb
from PIL import Image
from Unet_Resnet import ResNetUNet
import torchvision.transforms.functional as F

# ---------------------------
# 定义联合数据增强变换
# ---------------------------

class JointRandomCrop(object):
    def __init__(self, crop_size):
        """
        crop_size: 裁切尺寸，(h, w) 或 int（正方形）
        """
        if isinstance(crop_size, int):
            self.crop_size = (crop_size, crop_size)
        else:
            self.crop_size = crop_size
    
    def __call__(self, image, mask):
        # 使用 torchvision.transforms.RandomCrop.get_params 获取随机裁切参数
        i, j, h, w = transforms.RandomCrop.get_params(image, self.crop_size)
        image = F.crop(image, i, j, h, w)
        mask = F.crop(mask, i, j, h, w)
        return image, mask

class JointRandomAffine(object):
    def __init__(self, degrees, translate=None, scale=None, shear=None, fill=0):
        """
        degrees: 旋转角度范围，如 (-30, 30) 或单个数字（表示对称范围）
        translate: 平移比例范围，例如 (0.1, 0.1) 表示宽高各自最多平移 10%
        scale: 缩放范围，例如 (0.8, 1.2)
        shear: 剪切角度范围，例如 (-10, 10)
        fill: mask 填充值（旋转、平移后空缺部分），通常 mask 用 0 填充
        """
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.fill = fill
    
    def __call__(self, image, mask):
        # 随机采样旋转角度
        if isinstance(self.degrees, (tuple, list)):
            angle = random.uniform(self.degrees[0], self.degrees[1])
        else:
            angle = random.uniform(-self.degrees, self.degrees)
        # 随机采样平移参数
        if self.translate is not None:
            max_dx = self.translate[0] * image.size[0]
            max_dy = self.translate[1] * image.size[1]
            translations = (int(random.uniform(-max_dx, max_dx)), int(random.uniform(-max_dy, max_dy)))
        else:
            translations = (0, 0)
        # 随机采样缩放因子
        if self.scale is not None:
            scale = random.uniform(self.scale[0], self.scale[1])
        else:
            scale = 1.0
        # 随机采样剪切角度（这里只取单一角度，可扩展为左右剪切）
        if self.shear is not None:
            if isinstance(self.shear, (tuple, list)):
                shear = random.uniform(self.shear[0], self.shear[1])
            else:
                shear = self.shear
        else:
            shear = 0.0
        
        image = F.affine(image, angle=angle, translate=translations, scale=scale, shear=shear)
        mask = F.affine(mask, angle=angle, translate=translations, scale=scale, shear=shear, fill=self.fill)
        return image, mask

class JointCompose(object):
    def __init__(self, transforms_list):
        self.transforms = transforms_list
    
    def __call__(self, image, mask):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask

# ---------------------------
# 自定义数据集（加入联合变换）
# ---------------------------
class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, joint_transform=None, image_transform=None, mask_transform=None):
        """
        images_dir: 存放 RGB 图像的文件夹路径
        masks_dir: 存放 mask 图像的文件夹路径（假设文件名一致）
        joint_transform: 对图像和 mask 同时进行的几何数据增强（如随机裁切、旋转、平移）
        image_transform: 针对图像的其他预处理（如 ToTensor、Normalize）
        mask_transform: 针对 mask 的其他预处理（如 ToTensor）
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.joint_transform = joint_transform
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        
        self.image_files = sorted(os.listdir(images_dir))
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.images_dir, self.image_files[idx])
        mask_path = os.path.join(self.masks_dir, self.image_files[idx])  # 假设 mask 文件名与 image 相同
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # 灰度图
        
        # 先进行联合数据增强（几何变换，保持同步）
        if self.joint_transform:
            image, mask = self.joint_transform(image, mask)
        
        # 再分别进行各自的预处理（转换为 tensor 及归一化）
        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        
        # 二值化 mask（假设 mask 像素值本来为0或1之间）
        mask = (mask > 0.5).float()
        return image, mask

# ---------------------------
# 主函数
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', type=str, default='/home/hongjia/CV_H/dataset/train/train/images', help="训练图像文件夹路径")
    parser.add_argument('--masks_dir', type=str, default='/home/hongjia/CV_H/dataset/train/train/masks', help="mask 文件夹路径")
    parser.add_argument('--log_dir', type=str, default='./logs', help="日志保存路径")
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_enhanced_data', help="模型 checkpoint 保存路径")
    parser.add_argument('--batch_size', type=int, default=128, help="batch 大小")
    parser.add_argument('--num_epochs', type=int, default=1000, help="训练总轮数")
    parser.add_argument('--val_split', type=float, default=0.2, help="验证集比例（0~1之间）")
    parser.add_argument('--wandb_project', type=str, default='segmentation_project', help="wandb 项目名称")
    args = parser.parse_args()

    cudnn.enabled = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建 checkpoint 保存文件夹
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    
    # 初始化 wandb
    wandb.init(project=args.wandb_project, config=vars(args))
    
    # 定义联合数据增强（先进行随机裁切，再进行随机旋转、平移）
    joint_transform = JointCompose([
        JointRandomCrop((256, 256)),
        JointRandomAffine(degrees=(-30, 30), translate=(0.1, 0.1))
    ])
    
    # 单独的图像预处理：转换为 tensor 并归一化
    image_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # 单独的 mask 预处理：转换为 tensor
    mask_transforms = transforms.Compose([
        transforms.ToTensor()
    ])
    
    # 创建数据集
    full_dataset = SegmentationDataset(
        images_dir=args.images_dir,
        masks_dir=args.masks_dir,
        joint_transform=joint_transform,  # 联合几何变换
        image_transform=image_transforms,
        mask_transform=mask_transforms
    )
    
    # 划分训练集和验证集
    total_size = len(full_dataset)
    val_size = int(total_size * args.val_split)
    train_size = total_size - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # DataLoader
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # 初始化模型、损失函数和优化器
    model = ResNetUNet(out_channels=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    global_step = 0
    
    # 开始训练
    for epoch in range(args.num_epochs):
        model.train()
        train_loss = 0.0
        
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} - Training"):
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            wandb.log({"train_loss_batch": loss.item()}, step=global_step)
            global_step += 1
        
        train_loss /= len(train_dataset)
        wandb.log({"train_loss_epoch": train_loss, "epoch": epoch + 1})
        print(f"Epoch [{epoch+1}/{args.num_epochs}], Train Loss: {train_loss:.4f}")
        
        if (epoch + 1) % 50 == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} - Validation"):
                    images = images.to(device)
                    masks = masks.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    val_loss += loss.item() * images.size(0)
            
            val_loss /= len(val_dataset)
            wandb.log({"val_loss": val_loss, "epoch": epoch + 1})
            print(f"Epoch [{epoch+1}/{args.num_epochs}], Validation Loss: {val_loss:.4f}")
            
            checkpoint_path = os.path.join(args.checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")
    
    print("训练完成。")

if __name__ == '__main__':
    main()
