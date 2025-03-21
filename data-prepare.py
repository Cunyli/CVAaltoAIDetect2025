import os
import shutil
from tqdm import tqdm

# 定义路径
originals_dir = './data/originals'
images_dir = './data/images'
masks_dir = './data/masks'

# 输出路径（筛选后的数据）
filtered_images_dir = './data/filtered/images'
filtered_masks_dir = './data/filtered/masks'

# 创建输出目录（如果不存在）
os.makedirs(filtered_images_dir, exist_ok=True)
os.makedirs(filtered_masks_dir, exist_ok=True)

# 获取 originals 目录下的文件名（去掉后缀）
original_files = set([f.split('.')[0] for f in os.listdir(originals_dir)])

print(f"✅ 在 originals 目录下找到 {len(original_files)} 个文件")

# 遍历 images 和 masks，筛选出与 originals 匹配的文件
matched_count = 0
for file in tqdm(os.listdir(images_dir)):
    filename = file.split('.')[0]
    
    # 如果文件名存在于 originals 里
    if filename in original_files:
        # 复制到新的输出路径
        shutil.copy(os.path.join(images_dir, file), os.path.join(filtered_images_dir, file))
        shutil.copy(os.path.join(masks_dir, file), os.path.join(filtered_masks_dir, file))
        matched_count += 1

print(f"🎉 完成筛选！找到 {matched_count} 个与 originals 匹配的文件")
