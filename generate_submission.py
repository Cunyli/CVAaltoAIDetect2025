import os
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm

# ✅ 定义路径
# 输出的掩码文件夹
MASK_DIR = 'output/masks'
OUTPUT_CSV = 'submission.csv'

# ✅ 定义 RLE 编码函数
def mask2rle(img):
    """
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formatted
    """
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

# ✅ 生成 RLE 编码并保存为 CSV
def generate_submission():
    print("开始生成提交文件...")
    submission = []

    # 遍历输出的掩码文件
    for file_name in tqdm(os.listdir(MASK_DIR)):
        if file_name.endswith('.png'):
            image_id = file_name.split('.')[0]
            mask = cv2.imread(os.path.join(MASK_DIR, file_name), cv2.IMREAD_GRAYSCALE)

            # 将掩码转换为二值
            mask = (mask > 0).astype(np.uint8)

            # 转换为 RLE 格式
            rle = mask2rle(mask)

            # 添加到提交列表
            submission.append({'ImageId': image_id, 'EncodedPixels': rle})

    # 转换为 DataFrame
    df = pd.DataFrame(submission)

    # 保存为 CSV
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ 提交文件已保存为 {OUTPUT_CSV}")

# ✅ 执行生成
if __name__ == "__main__":
    generate_submission()
