import cv2
import numpy as np
import matplotlib.pyplot as plt

def create_high_frequency_image(width, height, frequency):
    """建立一個高頻率的線條圖案"""
    img = np.zeros((height, width), dtype=np.uint8)
    for x in range(width):
        # 使用 sin 函數來產生規律的亮暗線條
        val = (np.sin(x * 2 * np.pi * frequency / width) + 1) / 2 * 255
        img[:, x] = int(val)
    return img

def demonstrate_moire(original_img, downscale_factor):
    """
    演示莫列波紋：
    1. 降取樣 (縮小)
    2. 升取樣 (放大)
    """
    # 1. 降取樣 (模擬取樣不足)
    small_height = int(original_img.shape[0] / downscale_factor)
    small_width = int(original_img.shape[1] / downscale_factor)
    
    # 使用 cv2.resize 進行縮小
    downsampled_img = cv2.resize(original_img, (small_width, small_height), 
                                 interpolation=cv2.INTER_LINEAR)
    
    # 2. 升取樣 (放大回原尺寸，使用最近鄰插值法)
    # 這樣會讓 aliasing (混疊) 效果更明顯
    upsampled_img = cv2.resize(downsampled_img, (original_img.shape[1], original_img.shape[0]), 
                               interpolation=cv2.INTER_NEAREST)
    
    return upsampled_img

# --- 執行 ---

# 1. 建立一個 512x512 大小、高頻率的原始圖案
original_pattern = create_high_frequency_image(512, 512, frequency=100)

# 2. 演示莫列波紋 (縮小 4 倍再放大)
moire_artifact = demonstrate_moire(original_pattern, downscale_factor=4.0)

# --- 顯示結果 (使用 Matplotlib) ---import cv2
import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# 作業 1, Part 1: 莫列波紋 (Moiré Patterns) - 取樣不足
# ==========================================================

def create_high_frequency_image(width, height, frequency):
    """建立一個高頻率的線條圖案"""
    img = np.zeros((height, width), dtype=np.uint8)
    for x in range(width):
        val = (np.sin(x * 2 * np.pi * frequency / width) + 1) / 2 * 255
        img[:, x] = int(val)
    return img

def demonstrate_moire(original_img, downscale_factor):
    """演示莫列波紋：1. 降取樣 (縮小) 2. 升取樣 (放大)"""
    # 1. 降取樣 (模擬取樣不足)
    small_height = int(original_img.shape[0] / downscale_factor)
    small_width = int(original_img.shape[1] / downscale_factor)
    downsampled_img = cv2.resize(original_img, (small_width, small_height), 
                                 interpolation=cv2.INTER_LINEAR)
    
    # 2. 升取樣 (放大回原尺寸，使用最近鄰插值法)
    upsampled_img = cv2.resize(downsampled_img, (original_img.shape[1], original_img.shape[0]), 
                               interpolation=cv2.INTER_NEAREST)
    return upsampled_img

# --- 執行 Part 1 ---
print("正在產生 Part 1: 莫列波紋...")
original_pattern = create_high_frequency_image(512, 512, frequency=100)
moire_artifact = demonstrate_moire(original_pattern, downscale_factor=4.0)

# --- 顯示 Part 1 結果 ---
plt.figure(figsize=(12, 10)) # 調整畫布大小以容納兩組圖

plt.subplot(2, 2, 1) # 2x2 的第 1 張圖
plt.title("Original High-Freq Pattern")
plt.imshow(original_pattern, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 2) # 2x2 的第 2 張圖
plt.title(f"Moiré Pattern (Downscaled 4x)")
plt.imshow(moire_artifact, cmap='gray')
plt.axis('off')

# 儲存 Part 1 圖片
cv2.imwrite("original_pattern.png", original_pattern)
cv2.imwrite("moire_artifact.png", moire_artifact)


# ==========================================================
# 作業 1, Part 2: 偽輪廓 (False Contours) - 量化不足
# ==========================================================

def create_gradient_image(width, height):
    """建立一個灰階漸層影像"""
    img = np.zeros((height, width), dtype=np.uint8)
    # 建立一個從左到右的水平漸層 (0 -> 255)
    for x in range(width):
        img[:, x] = int(x / width * 255)
    return img

def demonstrate_false_contours(original_img, num_levels):
    """
    演示偽輪廓 (量化不足)
    num_levels: 你想要的灰階等級數量 (例如 4, 8, 16)
    """
    if num_levels <= 1 or num_levels > 255:
        return original_img
    
    # 計算每個量化區間的大小 (步長)
    # 確保我們使用 0-255 的完整範圍
    step = 256 // num_levels
    
    # 這是最關鍵的一步：
    # 1. (original_img // step): 將 0-255 對應到 0-(N-1) 的區間
    # 2. (... * step): 再將 0-(N-1) 的區間對應回 0-255 的量化值
    # 3. (+ step // 2): (可選) 加上半個步長，使量化值位於區間的"中間"
    quantized_img = (original_img // step) * step + (step // 2)
    
    # 確保值不會超過 255
    quantized_img[quantized_img > 255] = 255
    
    return quantized_img.astype(np.uint8)

# --- 執行 Part 2 ---
print("正在產生 Part 2: 偽輪廓...")
# 建立一個 512x256 的平滑漸層影像 (256 個灰階)
original_gradient = create_gradient_image(512, 256)

# 演示偽輪廓 (將 256 階 壓縮到 4 階)
N_levels = 4
false_contour_artifact = demonstrate_false_contours(original_gradient, num_levels=N_levels)

# --- 顯示 Part 2 結果 ---
plt.subplot(2, 2, 3) # 2x2 的第 3 張圖
plt.title("Original Gradient (256 Levels)")
plt.imshow(original_gradient, cmap='gray', vmin=0, vmax=255)
plt.axis('off')

plt.subplot(2, 2, 4) # 2x2 的第 4 張圖
plt.title(f"False Contours ({N_levels} Levels)")
plt.imshow(false_contour_artifact, cmap='gray', vmin=0, vmax=255)
plt.axis('off')

# 儲存 Part 2 圖片
cv2.imwrite("original_gradient.png", original_gradient)
cv2.imwrite("false_contour_artifact.png", false_contour_artifact)


# ==========================================================
# 最終顯示
# ==========================================================
plt.tight_layout() # 自動調整子圖間距
plt.show()

print("作業一 (HW1) 已完成，所有圖片已儲存。")
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title("Original High-Freq Pattern")
plt.imshow(original_pattern, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title(f"Moiré Pattern (Downscaled 4x)")
plt.imshow(moire_artifact, cmap='gray')
plt.axis('off')

plt.show()

# 儲存圖片
cv2.imwrite("original_pattern.png", original_pattern)
cv2.imwrite("moire_artifact.png", moire_artifact)