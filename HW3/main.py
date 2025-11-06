import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- 1. 建立一個範例影像 (代替讀取圖片) ---
img_height, img_width = 400, 400
img = np.zeros((img_height, img_width, 3), dtype=np.uint8)
img.fill(30) # 灰色背景

# 定義我們要 "校正" 的原始四邊形 (來源點)
# 順序：左上, 右上, 左下, 右下
src_pts = np.float32([
    [70, 80],   # 左上
    [330, 60],  # 右上
    [50, 350],  # 左下
    [350, 360]  # 右下
])

# 在範例影像上畫出這個四邊形
cv2.polylines(img, [src_pts.astype(np.int32)], isClosed=True, color=(0, 255, 255), thickness=2)
cv2.putText(img, "SKEWED TEXT", (80, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
print("已建立範例原始影像...")

# --- 2. 定義目標點 ---
# 我們希望將上面的 src_pts 拉伸到一個 300x200 的完美矩形
out_width, out_height = 300, 200
dst_pts = np.float32([
    [0, 0],
    [out_width - 1, 0],
    [0, out_height - 1],
    [out_width - 1, out_height - 1]
])

# --- 3. 手動建立 8 參數的 A h = b 系統 ---
A = np.zeros((8, 8))
b = np.zeros((8, 1))

for i in range(4):
    x, y = src_pts[i]
    xp, yp = dst_pts[i] 
    
    A[2*i] = [x, y, 1, 0, 0, 0, -x*xp, -y*xp]
    b[2*i] = xp
    
    A[2*i + 1] = [0, 0, 0, x, y, 1, -x*yp, -y*yp]
    b[2*i + 1] = yp

print("已建立 A (8x8) 和 b (8x1) 矩陣...")

# --- 4. 求解 8 參數 h ---
try:
    h = np.linalg.solve(A, b)
except np.linalg.LinAlgError:
    print("錯誤：A 矩陣是奇異矩陣 (Singular)，無法求解。")
    exit()

H_matrix_manual = np.append(h, 1).reshape((3, 3))

print("已解出 8 參數，並重建 3x3 矩陣 H：")
print(H_matrix_manual)

# --- 5. 驗證我們的 H 和 OpenCV 的是否一致 ---
H_opencv = cv2.getPerspectiveTransform(src_pts, dst_pts)

print("\nOpenCV 算出的 H 矩陣 (用於比較)：")
print(H_opencv)


# --- 6. 套用我們手動算出的 H 矩陣來進行透視轉換 ---
warped_img = cv2.warpPerspective(img, H_matrix_manual, (out_width, out_height))

print("已使用手動計算的 H 矩陣完成影像轉換！")

# --- 7. 顯示結果 ---
plt.figure(figsize=(12, 6))

# 顯示原始影像
plt.subplot(1, 2, 1)
plt.title("Original Image (with source points)")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # 轉為 RGB 供 plt 顯示
for pt in src_pts:
    cv2.circle(img_rgb, tuple(pt.astype(int)), 5, (255, 0, 0), -1) # 標出紅點
plt.imshow(img_rgb)
plt.axis('off')

# 顯示校正後的影像
plt.subplot(1, 2, 2)
plt.title("Warped Image (using manual H matrix)")
warped_img_rgb = cv2.cvtColor(warped_img, cv2.COLOR_BGR2RGB)
plt.imshow(warped_img_rgb)
plt.axis('off')

plt.tight_layout()
plt.show()

# --- 8. 儲存結果 ---
cv2.imwrite("hw3_original.png", img)
cv2.imwrite("hw3_warped.png", warped_img)

print("HW3 影像已儲存。")