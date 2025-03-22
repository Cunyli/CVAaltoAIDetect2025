#!/usr/bin/env python3
# train.py

import os
import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms.functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import torchvision.utils as vutils

import wandb  # 若不需要 wandb，可去掉相关代码

# 从你的模型文件中导入网络
from Unet_Resnet import ResNetUNet

def unnormalize(tensor, mean, std):
    """
    对一个 4D Tensor[B, C, H, W] 进行反归一化操作，恢复到 [0,1] 左右。
    注意：此函数会原地修改 tensor 的数值。
    """
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)  # t = t * s + m
    return torch.clamp(tensor, 0.0, 1.0)  # 再夹紧到 [0,1]


# =============== 1. 定义多图像联合变换 ===============
class JointRandomCropN:
    """对传入的 N 张 PIL 图像执行相同的 RandomCrop。"""
    def __init__(self, crop_size):
        if isinstance(crop_size, int):
            self.crop_size = (crop_size, crop_size)
        else:
            self.crop_size = crop_size
    
    def __call__(self, *imgs):
        # 从第一张图像中获取随机裁切参数
        i, j, h, w = transforms.RandomCrop.get_params(imgs[0], self.crop_size)
        results = []
        for im in imgs:
            results.append(F.crop(im, i, j, h, w))
        return tuple(results)


class JointRandomAffineN:
    """对传入的 N 张 PIL 图像执行相同的 RandomAffine。"""
    def __init__(self, degrees, translate=None, scale=None, shear=None, fill=0):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.fill = fill
    
    def __call__(self, *imgs):
        # 1) 随机采样旋转角度
        if isinstance(self.degrees, (tuple, list)):
            angle = random.uniform(self.degrees[0], self.degrees[1])
        else:
            angle = random.uniform(-self.degrees, self.degrees)
        
        # 2) 平移
        if self.translate is not None:
            max_dx = self.translate[0] * imgs[0].width
            max_dy = self.translate[1] * imgs[0].height
            translations = (int(random.uniform(-max_dx, max_dx)),
                            int(random.uniform(-max_dy, max_dy)))
        else:
            translations = (0, 0)
        
        # 3) 缩放
        if self.scale is not None:
            scale = random.uniform(self.scale[0], self.scale[1])
        else:
            scale = 1.0
        
        # 4) 剪切
        if self.shear is not None:
            if isinstance(self.shear, (tuple, list)):
                shear = random.uniform(self.shear[0], self.shear[1])
            else:
                shear = self.shear
        else:
            shear = 0.0
        
        # 对 N 张图像分别执行相同的 affine 变换
        results = []
        for im in imgs:
            # 如果有 mask，可以在这里区分 fill 值；此处统一处理
            transformed = F.affine(
                im,
                angle=angle,
                translate=translations,
                scale=scale,
                shear=shear,
                fill=self.fill
            )
            results.append(transformed)
        return tuple(results)


class JointComposeN:
    """依次对传入的 N 张图像执行多个联合变换。"""
    def __init__(self, transforms_list):
        self.transforms = transforms_list
    
    def __call__(self, *imgs):
        for t in self.transforms:
            imgs = t(*imgs)  # 每个 t 都返回一个 tuple
        return imgs


# =============== 2. 定义数据集，加载 images & originals ===============
class PairedDataset(Dataset):
    """
    同时加载:
      - images_dir 下的图像 (输入)
      - originals_dir 下的图像 (目标，做 MSE)
    并对二者进行相同的数据增强。
    """
    def __init__(self,
                 images_dir,
                 originals_dir,
                 joint_transform=None,
                 image_transform=None,
                 original_transform=None):
        self.images_dir = images_dir
        self.originals_dir = originals_dir
        self.joint_transform = joint_transform
        self.image_transform = image_transform
        self.original_transform = original_transform
        
        # 假设 images_dir 和 originals_dir 里文件名对应
        self.image_files = sorted(os.listdir(images_dir))
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.images_dir, image_name)
        original_path = os.path.join(self.originals_dir, image_name)
        
        image = Image.open(image_path).convert('RGB')
        original = Image.open(original_path).convert('RGB')
        
        # 1) 先进行联合变换 (随机裁切、随机旋转等)
        if self.joint_transform:
            image, original = self.joint_transform(image, original)
        
        # 2) 分别对 image / original 做单独变换 (ToTensor, Normalize等)
        if self.image_transform:
            image = self.image_transform(image)
        if self.original_transform:
            original = self.original_transform(original)
        
        return image, original


# =============== 3. 训练逻辑 ===============
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', type=str, default='dataset/filtered/images',
                        help="输入图像文件夹路径")
    parser.add_argument('--originals_dir', type=str, default='dataset/filtered/originals',
                        help="目标图像文件夹路径")
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_mse',
                        help="模型 checkpoint 保存路径")
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help="日志保存路径（若需要）")
    parser.add_argument('--wandb_project', type=str, default='segmentation_project',
                        help="wandb 项目名称")
    parser.add_argument('--val_split', type=float, default=0.2,
                        help="验证集比例（0~1之间）")
    parser.add_argument('--batch_size', type=int, default=64,
                        help="batch 大小")
    parser.add_argument('--num_epochs', type=int, default=500,
                        help="训练总轮数")
    parser.add_argument('--lr', type=float, default=1e-3,
                        help="学习率")
    parser.add_argument('--wandb_run_name', type=str, default=None,
                        help="wandb 运行名称（可自定义）")

    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True
    
    # 初始化 wandb (可选)
    # 当你初始化 wandb 时，指定 name=args.wandb_run_name
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,  # <-- 在这里传入自定义的 run 名称
        config=vars(args)
    )
    
    # 3.1 定义联合数据增强
    joint_transform = JointComposeN([
        JointRandomCropN((256, 256)),
        JointRandomAffineN(degrees=(-30, 30), translate=(0.1, 0.1))
    ])
    
    # 3.2 定义单独的图像预处理
    #     通常对 input 和 original 做相同的 Normalize，保证数值范围一致
    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    original_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # 3.3 构建完整数据集
    full_dataset = PairedDataset(
        images_dir=args.images_dir,
        originals_dir=args.originals_dir,
        joint_transform=joint_transform,
        image_transform=image_transform,
        original_transform=original_transform
    )
    
    # 3.4 划分训练集和验证集
    total_size = len(full_dataset)
    val_size = int(total_size * args.val_split)
    train_size = total_size - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # 3.5 构建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # 3.6 初始化模型、损失函数和优化器
    model = ResNetUNet(out_channels=3).to(device)
    criterion = nn.MSELoss()  # 与 originals 做 MSE
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # 3.7 训练循环
    global_step = 0
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    
    for epoch in range(args.num_epochs):
        model.train()
        train_loss = 0.0
        
        for images, originals in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [Train]"):
            images = images.to(device)
            originals = originals.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)  # 期望输出 shape: [B, 3, 256, 256]
            loss = criterion(outputs, originals)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            
            wandb.log({"train_loss_batch": loss.item()}, step=global_step)
            global_step += 1
        
        train_loss_epoch = train_loss / train_size
        wandb.log({"train_loss_epoch": train_loss_epoch, "epoch": epoch + 1})
        print(f"[Epoch {epoch+1}/{args.num_epochs}] Train Loss: {train_loss_epoch:.4f}")
        
        # 每隔若干个 epoch 做一次验证，并保存 checkpoint

        if (epoch + 1) % 1 == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for i, (images, originals) in enumerate(tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [Val]")):
                    images = images.to(device)
                    originals = originals.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, originals)
                    val_loss += loss.item() * images.size(0)
                    
                    # ---- 只在第一个 batch 可视化 8 张图像并上传到 wandb ----
                    if i == 0:
                        num_samples = min(8, images.size(0))
                        
                        # 克隆前 num_samples 张图像，避免直接改动原始 tensor
                        img_sample = images[:num_samples].clone()
                        out_sample = outputs[:num_samples].clone()
                        orig_sample = originals[:num_samples].clone()
                        
                        # 如果你的输入和输出也做了同样的 Normalize，可反归一化
                        # 若不需要，可直接跳过 unnormalize
                        mean = [0.485, 0.456, 0.406]
                        std = [0.229, 0.224, 0.225]
                        img_sample = unnormalize(img_sample, mean, std)
                        out_sample = unnormalize(out_sample, mean, std)
                        orig_sample = unnormalize(orig_sample, mean, std)
                        
                        # 拼接每个样本的图像（横向拼接：输入、输出、标签）
                        combined = torch.cat([img_sample, out_sample, orig_sample], dim=3)
                        # 这里 combined 的形状变为 [num_samples, 3, H, 3*W]

                        # 使用 make_grid 将多张对比图排成一张大图，nrow=1 表示每行只放一张（即每个样本占一行）
                        combined_grid = vutils.make_grid(combined, nrow=1)

                        # 上传到 wandb 作为单张对比图
                        wandb.log({
                            "Val_Comparison": wandb.Image(combined_grid, caption="Input | Output | Original")
                        })

            
            val_loss_epoch = val_loss / val_size
            wandb.log({"val_loss": val_loss_epoch, "epoch": epoch + 1})
            print(f"[Epoch {epoch+1}/{args.num_epochs}] Val Loss: {val_loss_epoch:.4f}")
            
            # (可选) 保存 checkpoint
            ckpt_path = os.path.join(args.checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Checkpoint saved at {ckpt_path}")

    
    print("训练完成。")


if __name__ == '__main__':
    main()
