import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import os
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torch.backends.cudnn as cudnn
from torchvision import transforms
from tqdm import tqdm
import wandb
import timm


class MultiBranchDataset(Dataset):
    def __init__(self, 
                 images_dir, 
                 masks_dir, 
                 originals_dir,
                 image_transform=None, 
                 mask_transform=None,
                 orig_transform=None):
        """
        images_dir: 存放 AI 修改图像的文件夹
        masks_dir:  存放 mask 的文件夹
        originals_dir: 存放原图的文件夹
        image_transform: 对 AI 修改图的预处理
        mask_transform: 对 mask 的预处理
        orig_transform: 对原图的预处理
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.originals_dir = originals_dir
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.orig_transform = orig_transform
        
        self.image_files = sorted(os.listdir(images_dir))  # 假设文件名与 masks、originals 对应

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]
        # 路径
        image_path = os.path.join(self.images_dir, filename)
        mask_path = os.path.join(self.masks_dir, filename)
        orig_path = os.path.join(self.originals_dir, filename)

        # 读取
        ai_img = Image.open(image_path).convert('RGB')  # AI 修改图 (输入)
        mask = Image.open(mask_path).convert('L')       # mask
        orig_img = Image.open(orig_path).convert('RGB') # 原图 (重构目标)

        # 预处理
        if self.image_transform:
            ai_img = self.image_transform(ai_img)
        if self.mask_transform:
            mask = self.mask_transform(mask)
            # 将 mask 二值化（假设原始 mask 像素值为0或255）
            mask = (mask > 0.5).float()
        if self.orig_transform:
            orig_img = self.orig_transform(orig_img)

        return ai_img, mask, orig_img

class ViT_UNet(nn.Module):
    def __init__(self, 
                 out_channels=4, 
                 img_size=256, 
                 patch_size=16, 
                 embed_dim=768, 
                 num_transformer_layers=2):
        """
        out_channels=4: 1 通道 mask + 3 通道重构
        """
        super(ViT_UNet, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # 1. 预训练的 ViT
        self.vit = timm.create_model('vit_base_patch16_224', 
                                     pretrained=True, 
                                     img_size=img_size)
        self.num_patches = (img_size // patch_size) ** 2
        
        # 2. Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=8)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, 
                                                         num_layers=num_transformer_layers)
        self.query_embed = nn.Parameter(torch.randn(self.num_patches, embed_dim))
        
        # 3. 卷积上采样
        self.upsample_conv = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        # 1. ViT encoder
        tokens = self.vit.forward_features(x)  # [B, 1+N, embed_dim]
        tokens = tokens[:, 1:, :]              # 去掉 class token -> [B, N, embed_dim]

        # 2. Transformer decoder
        memory = tokens.transpose(0, 1)  # [N, B, embed_dim]
        queries = self.query_embed.unsqueeze(1).repeat(1, x.size(0), 1)  # [N, B, embed_dim]
        decoded = self.transformer_decoder(queries, memory)  # [N, B, embed_dim]
        decoded = decoded.transpose(0, 1)                    # [B, N, embed_dim]

        # 3. reshape -> [B, embed_dim, H, W]
        H = W = self.img_size // self.patch_size
        feat_map = decoded.transpose(1, 2).contiguous().view(x.size(0), self.embed_dim, H, W)

        # 4. upsample -> [B, 4, 256, 256]
        x_up = self.upsample_conv(feat_map)
        out = self.final_conv(x_up)
        return out

# ---------- 引入上面定义的模型和数据集 ----------
# from your_code import ViT_UNet, MultiBranchDataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', type=str, default='/home/hongjia/CV_H/dataset/filtered/images',
                        help="AI修改后图像路径")
    parser.add_argument('--masks_dir', type=str, default='/home/hongjia/CV_H/dataset/filtered/masks',
                        help="mask 文件夹路径")
    parser.add_argument('--originals_dir', type=str, default='/home/hongjia/CV_H/dataset/filtered/originals',
                        help="原图文件夹路径")
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help="模型 checkpoint 保存路径")
    parser.add_argument('--batch_size', type=int, default=12, help="batch 大小")
    parser.add_argument('--num_epochs', type=int, default=50, help="训练总轮数")
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
    
    # 预处理
    image_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    mask_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    orig_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        # 若也想对原图做同样的归一化，可加上 Normalize
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225])
    ])
    
    # 创建数据集
    full_dataset = MultiBranchDataset(
        images_dir=args.images_dir,
        masks_dir=args.masks_dir,
        originals_dir=args.originals_dir,
        image_transform=image_transforms,
        mask_transform=mask_transforms,
        orig_transform=orig_transforms
    )
    
    # 划分训练集和验证集
    total_size = len(full_dataset)
    val_size = int(total_size * args.val_split)
    train_size = total_size - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    
    # 初始化模型
    model = ViT_UNet(out_channels=4, img_size=256, patch_size=16, embed_dim=768, num_transformer_layers=2)
    model = model.to(device)

    # 损失函数：BCE 用于 mask，MSE 用于重构
    criterion_mask = nn.BCEWithLogitsLoss()
    criterion_recon = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    global_step = 0
    
    for epoch in range(args.num_epochs):
        model.train()
        train_loss = 0.0
        
        for ai_img, mask, orig_img in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} - Training"):
            ai_img = ai_img.to(device)    # [B,3,256,256]
            mask   = mask.to(device)      # [B,1,256,256]
            orig_img = orig_img.to(device)# [B,3,256,256]
            
            optimizer.zero_grad()
            outputs = model(ai_img)       # [B,4,256,256]
            
            # 第 1 通道 -> mask_logits
            mask_logits = outputs[:, 0:1, :, :]
            # 后 3 通道 -> 重构图
            recon = outputs[:, 1:4, :, :]
            
            # 计算损失
            loss_mask = criterion_mask(mask_logits, mask)
            loss_recon = criterion_recon(recon, orig_img)
            loss = loss_mask + loss_recon
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * ai_img.size(0)
            
            wandb.log({
                "train_loss_batch": loss.item(),
                "train_loss_mask": loss_mask.item(),
                "train_loss_recon": loss_recon.item()
            }, step=global_step)
            global_step += 1
        
        train_loss /= len(train_dataset)
        wandb.log({"train_loss_epoch": train_loss, "epoch": epoch + 1})
        print(f"Epoch [{epoch+1}/{args.num_epochs}], Train Loss: {train_loss:.4f}")
        
        # 每5个 epoch 验证并保存
        if (epoch + 1) % 5 == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for ai_img, mask, orig_img in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} - Validation"):
                    ai_img = ai_img.to(device)
                    mask   = mask.to(device)
                    orig_img = orig_img.to(device)
                    
                    outputs = model(ai_img)
                    mask_logits = outputs[:, 0:1, :, :]
                    recon = outputs[:, 1:4, :, :]
                    
                    loss_mask = criterion_mask(mask_logits, mask)
                    loss_recon = criterion_recon(recon, orig_img)
                    loss = loss_mask + loss_recon
                    val_loss += loss.item() * ai_img.size(0)
            
            val_loss /= len(val_dataset)
            wandb.log({"val_loss": val_loss, "epoch": epoch + 1})
            print(f"Epoch [{epoch+1}/{args.num_epochs}], Validation Loss: {val_loss:.4f}")
            
            checkpoint_path = os.path.join(args.checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")
    
    print("训练完成。")


if __name__ == '__main__':
    main()
