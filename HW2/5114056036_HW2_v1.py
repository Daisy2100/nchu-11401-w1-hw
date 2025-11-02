import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- 1. 建立兩張 300x300 的空白二元影像 ---
img_size = (300, 300)
height, width = img_size
img1 = np.zeros(img_size, dtype=np.uint8)
img2 = np.zeros(img_size, dtype=np.uint8)

# --- 2. 在影像上繪製形狀 (這部分仍使用 cv2 輔助) ---
cv2.rectangle(img1, (50, 50), (200, 200), 255, -1) 
cv2.circle(img2, (175, 175), 100, 255, -1) 

print("已建立 img1 (矩形) 和 img2 (圓形)...")

# --- 3. 手動實現邏輯運算 (不使用 cv2.bitwise_... 函式) ---

# 建立兩個空白的畫布來存放結果
img_and_manual = np.zeros(img_size, dtype=np.uint8)
img_union_manual = np.zeros(img_size, dtype=np.uint8)

print("正在手動執行 AND 和 Union (OR) 運算...")

# ==================== 核心邏輯開始 ====================
# 使用巢狀 for 迴圈逐一檢查每個像素 (y 代表高, x 代表寬)
for y in range(height):
    for x in range(width):
        # 讀取兩張圖片在 (y, x) 位置的像素值 (0 或 255)
        pixel_1 = img1[y, x]
        pixel_2 = img2[y, x]
        
        # --- 邏輯 AND (交集) ---
        # 條件：只有當兩個像素 *都* 是 255 (白色) 時，結果才是 255
        if pixel_1 == 255 and pixel_2 == 255:
            img_and_manual[y, x] = 255
        # (在其他情況下，它會保持預設的 0 (黑色))
            
        # --- 邏輯 Union (聯集 / OR) ---
        # 條件：只要 *任何一個* 像素是 255 (白色) 時，結果就是 255
        if pixel_1 == 255 or pixel_2 == 255:
            img_union_manual[y, x] = 255
        # (在其他情況下，它會保持預設的 0 (黑色))
# ==================== 核心邏輯結束 ====================

print("運算完成！")

# --- 4. 顯示結果 (使用 matplotlib) ---
plt.figure(figsize=(10, 10))

plt.subplot(2, 2, 1)
plt.title("Image 1 (Rectangle)")
plt.imshow(img1, cmap='gray')
plt.axis('off') 

plt.subplot(2, 2, 2)
plt.title("Image 2 (Circle)")
plt.imshow(img2, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.title("Manual Logical AND (Intersection)")
plt.imshow(img_and_manual, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.title("Manual Logical Union (OR)")
plt.imshow(img_union_manual, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()

# --- 5. 儲存結果影像 (使用 cv2) ---
cv2.imwrite("hw2_img1_rect.png", img1)
cv2.imwrite("hw2_img2_circle.png", img2)
cv2.imwrite("hw2_manual_and.png", img_and_manual)
cv2.imwrite("hw2_manual_union.png", img_union_manual)

print("HW2 (手動版) 影像已儲存。")