import cv2
import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# HW4: 影像銳化 (Image Sharpening)
# 要求：
# 1. 基於拉普拉斯運算子實作影像銳化
# 2. 基於梯度遮罩實作影像銳化
# 3. 手動實作梯度計算和卷積運算
# 4. 包含適當的零填充
# 5. 禁止使用內建卷積 API
# ==========================================================

def manual_padding(image, pad_size, mode='zero'):
    """
    手動實作零填充 (Zero-padding)
    
    參數:
        image: 輸入影像
        pad_size: 填充大小
        mode: 填充模式 ('zero' 或 'replicate')
    
    返回:
        填充後的影像
    """
    h, w = image.shape
    padded = np.zeros((h + 2*pad_size, w + 2*pad_size), dtype=image.dtype)
    
    if mode == 'zero':
        # 零填充
        padded[pad_size:h+pad_size, pad_size:w+pad_size] = image
    elif mode == 'replicate':
        # 邊緣複製填充
        padded[pad_size:h+pad_size, pad_size:w+pad_size] = image
        # 填充上下左右邊界
        padded[:pad_size, pad_size:w+pad_size] = image[0:1, :]
        padded[h+pad_size:, pad_size:w+pad_size] = image[-1:, :]
        padded[pad_size:h+pad_size, :pad_size] = image[:, 0:1]
        padded[pad_size:h+pad_size, w+pad_size:] = image[:, -1:]
        # 填充四個角
        padded[:pad_size, :pad_size] = image[0, 0]
        padded[:pad_size, w+pad_size:] = image[0, -1]
        padded[h+pad_size:, :pad_size] = image[-1, 0]
        padded[h+pad_size:, w+pad_size:] = image[-1, -1]
    
    return padded


def manual_convolution(image, kernel):
    """
    手動實作 2D 卷積運算 (不使用內建 API)
    
    參數:
        image: 輸入影像
        kernel: 卷積核 (必須是奇數大小)
    
    返回:
        卷積結果
    """
    # 取得影像和卷積核的尺寸
    img_h, img_w = image.shape
    kernel_h, kernel_w = kernel.shape
    
    # 計算填充大小
    pad_h = kernel_h // 2
    pad_w = kernel_w // 2
    
    # 進行零填充
    padded_img = manual_padding(image, max(pad_h, pad_w), mode='replicate')
    
    # 建立輸出影像
    output = np.zeros((img_h, img_w), dtype=np.float64)
    
    # 手動執行卷積運算
    print(f"正在執行卷積運算 (影像大小: {img_h}x{img_w}, 核大小: {kernel_h}x{kernel_w})...")
    
    for i in range(img_h):
        for j in range(img_w):
            # 提取當前區域
            region = padded_img[i:i+kernel_h, j:j+kernel_w]
            
            # 執行卷積：元素相乘後加總
            conv_sum = 0.0
            for ki in range(kernel_h):
                for kj in range(kernel_w):
                    conv_sum += region[ki, kj] * kernel[ki, kj]
            
            output[i, j] = conv_sum
    
    return output


def laplacian_sharpening(image):
    """
    使用拉普拉斯運算子進行影像銳化
    
    拉普拉斯核:
    [ 0 -1  0]
    [-1  4 -1]
    [ 0 -1  0]
    
    銳化公式: sharpened = original - laplacian
    """
    print("\n=== 拉普拉斯銳化 ===")
    
    # 定義拉普拉斯核 (4-連通)
    laplacian_kernel = np.array([
        [ 0, -1,  0],
        [-1,  4, -1],
        [ 0, -1,  0]
    ], dtype=np.float64)
    
    print("使用的拉普拉斯核:")
    print(laplacian_kernel)
    
    # 手動執行卷積
    laplacian_result = manual_convolution(image.astype(np.float64), laplacian_kernel)
    
    # 銳化: 原始影像 - 拉普拉斯結果
    sharpened = image.astype(np.float64) - laplacian_result
    
    # 限制在 [0, 255] 範圍內
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    
    return sharpened, laplacian_result


def gradient_based_sharpening(image):
    """
    使用 Sobel 梯度遮罩進行影像銳化
    
    Sobel X 核:          Sobel Y 核:
    [-1  0  1]          [-1 -2 -1]
    [-2  0  2]          [ 0  0  0]
    [-1  0  1]          [ 1  2  1]
    
    梯度強度: |G| = sqrt(Gx^2 + Gy^2)
    銳化: sharpened = original + gradient_magnitude
    """
    print("\n=== 梯度遮罩銳化 (Sobel) ===")
    
    # 定義 Sobel 梯度核
    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float64)
    
    sobel_y = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ], dtype=np.float64)
    
    print("Sobel X 核:")
    print(sobel_x)
    print("\nSobel Y 核:")
    print(sobel_y)
    
    # 手動計算 X 和 Y 方向的梯度
    img_float = image.astype(np.float64)
    gradient_x = manual_convolution(img_float, sobel_x)
    gradient_y = manual_convolution(img_float, sobel_y)
    
    # 計算梯度強度: |G| = sqrt(Gx^2 + Gy^2)
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    
    # 正規化梯度強度到 [0, 255]
    gradient_magnitude_normalized = (gradient_magnitude / gradient_magnitude.max() * 255).astype(np.uint8)
    
    # 銳化: 原始影像 + 梯度強度 (加權 0.5 避免過度銳化)
    sharpened = image.astype(np.float64) + gradient_magnitude * 0.5
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    
    return sharpened, gradient_magnitude_normalized, gradient_x, gradient_y


# ==========================================================
# 主程式
# ==========================================================

print("HW4: 影像銳化技術")
print("=" * 60)

# 讀取影像
img = cv2.imread('origin.jpg', cv2.IMREAD_GRAYSCALE)

if img is None:
    print("錯誤：無法讀取 origin.jpg")
    print("請確認檔案存在於當前目錄")
    exit()

print(f"成功讀取影像，大小: {img.shape[1]}x{img.shape[0]}")

# 1. 拉普拉斯運算子銳化
laplacian_sharp, laplacian_edges = laplacian_sharpening(img)

# 2. 梯度遮罩銳化
gradient_sharp, gradient_mag, gradient_x, gradient_y = gradient_based_sharpening(img)

print("\n銳化處理完成！")

# ==========================================================
# 顯示結果
# ==========================================================

# 圖 1: 拉普拉斯運算子銳化
plt.figure(figsize=(16, 5))

plt.subplot(1, 4, 1)
plt.title("Original Image", fontsize=11, fontweight='bold')
plt.imshow(img, cmap='gray')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.title("Laplacian Edges", fontsize=11)
plt.imshow(np.abs(laplacian_edges), cmap='gray')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.title("Laplacian Sharpened", fontsize=11, fontweight='bold')
plt.imshow(laplacian_sharp, cmap='gray')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.title("Original vs Sharpened", fontsize=11)
# 左右對比
comparison = np.hstack([img, laplacian_sharp])
plt.imshow(comparison, cmap='gray')
plt.axis('off')

plt.suptitle("Laplacian Operator Based Sharpening", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('hw4_laplacian_sharpening.png', dpi=150, bbox_inches='tight')
print("\n已儲存: hw4_laplacian_sharpening.png")

# 圖 2: 梯度遮罩銳化
plt.figure(figsize=(20, 8))

plt.subplot(2, 4, 1)
plt.title("Original Image", fontsize=11, fontweight='bold')
plt.imshow(img, cmap='gray')
plt.axis('off')

plt.subplot(2, 4, 2)
plt.title("Gradient X (Sobel)", fontsize=11)
plt.imshow(np.abs(gradient_x), cmap='gray')
plt.axis('off')

plt.subplot(2, 4, 3)
plt.title("Gradient Y (Sobel)", fontsize=11)
plt.imshow(np.abs(gradient_y), cmap='gray')
plt.axis('off')

plt.subplot(2, 4, 4)
plt.title("Gradient Magnitude", fontsize=11)
plt.imshow(gradient_mag, cmap='gray')
plt.axis('off')

plt.subplot(2, 4, 5)
plt.title("Gradient Sharpened", fontsize=11, fontweight='bold')
plt.imshow(gradient_sharp, cmap='gray')
plt.axis('off')

plt.subplot(2, 4, 6)
plt.title("Original vs Sharpened", fontsize=11)
comparison = np.hstack([img, gradient_sharp])
plt.imshow(comparison, cmap='gray')
plt.axis('off')

# 放大顯示細節差異
h, w = img.shape
crop_y, crop_x = h//3, w//3
crop_size = 150
plt.subplot(2, 4, 7)
plt.title("Original (Detail)", fontsize=11)
plt.imshow(img[crop_y:crop_y+crop_size, crop_x:crop_x+crop_size], cmap='gray')
plt.axis('off')

plt.subplot(2, 4, 8)
plt.title("Sharpened (Detail)", fontsize=11)
plt.imshow(gradient_sharp[crop_y:crop_y+crop_size, crop_x:crop_x+crop_size], cmap='gray')
plt.axis('off')

plt.suptitle("Gradient-Based (Sobel) Sharpening", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('hw4_gradient_sharpening.png', dpi=150, bbox_inches='tight')
print("已儲存: hw4_gradient_sharpening.png")

# 圖 3: 兩種方法總比較
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title("Original Image", fontsize=12, fontweight='bold')
plt.imshow(img, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Laplacian Sharpened", fontsize=12, fontweight='bold')
plt.imshow(laplacian_sharp, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Gradient (Sobel) Sharpened", fontsize=12, fontweight='bold')
plt.imshow(gradient_sharp, cmap='gray')
plt.axis('off')

plt.suptitle("HW4: Image Sharpening Comparison", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('hw4_comparison.png', dpi=150, bbox_inches='tight')
print("已儲存: hw4_comparison.png")

plt.show()

# ==========================================================
# 儲存個別結果
# ==========================================================

cv2.imwrite("hw4_original.png", img)
cv2.imwrite("hw4_laplacian_edges.png", np.abs(laplacian_edges).astype(np.uint8))
cv2.imwrite("hw4_laplacian_sharpened.png", laplacian_sharp)
cv2.imwrite("hw4_gradient_magnitude.png", gradient_mag)
cv2.imwrite("hw4_gradient_sharpened.png", gradient_sharp)

print("已儲存: hw4_original.png")
print("已儲存: hw4_laplacian_edges.png")
print("已儲存: hw4_laplacian_sharpened.png")
print("已儲存: hw4_gradient_magnitude.png")
print("已儲存: hw4_gradient_sharpened.png")

print("\n" + "=" * 60)
print("HW4 影像銳化完成！")
print("=" * 60)
print("\n輸出檔案:")
print("  1. hw4_laplacian_sharpening.png  - 拉普拉斯銳化詳細過程")
print("  2. hw4_gradient_sharpening.png   - 梯度銳化詳細過程")
print("  3. hw4_comparison.png            - 兩種方法比較")
print("  4. hw4_*_sharpened.png           - 個別銳化結果")
print("=" * 60)
