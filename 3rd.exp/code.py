import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 创建result文件夹
result_dir = "result"
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# 读取图像
img_a = cv2.imread("img/a.jpg")
img_b = cv2.imread("img/b.jpg")

# 将图像转换为灰度图
gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)

# 初始化SIFT检测器并提取特征
sift = cv2.SIFT_create()
kp_a, des_a = sift.detectAndCompute(gray_a, None)  # 图像的关键点和描述符
kp_b, des_b = sift.detectAndCompute(gray_b, None)  # 图像的关键点和描述符

# 使用FLANN匹配器进行特征匹配
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)  # 检查次数越多，匹配越准确但速度越慢
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des_a, des_b, k=2)  # k=2表示每个特征点返回2个最佳匹配

# 应用Lowe's比率测试筛选优质匹配点
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:  # 比率阈值通常取0.7-0.8
        good_matches.append(m)

# 绘制匹配的SIFT关键点
matched_keypoints_img = cv2.drawMatches(
    img_a, kp_a, img_b, kp_b, good_matches,
    None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

# 提取匹配点的坐标
src_pts = np.float32([kp_b[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)  # 图像的关键点
dst_pts = np.float32([kp_a[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)  # 图像的关键点

# 使用RANSAC算法估计单应矩阵（透视变换矩阵）
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# 获取输入图像尺寸
h_a, w_a = img_a.shape[:2]
h_b, w_b = img_b.shape[:2]

# 计算图像变换后的四个角坐标
pts = np.float32([[0, 0], [0, h_b], [w_b, h_b], [w_b, 0]]).reshape(-1, 1, 2)
dst_corners = cv2.perspectiveTransform(pts, H)

# 确定拼接后图像的最终尺寸（包含所有像素）
all_corners = np.concatenate([dst_corners, np.float32([[0,0], [w_a,0], [w_a,h_a], [0,h_a]]).reshape(-1,1,2)], axis=0)
[x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() + 0.5)
[x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

# 创建平移矩阵，确保所有像素都在可视区域内
translation_matrix = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float32)

# 对图像进行透视变换和平移
fus_img = cv2.warpPerspective(
    img_b,
    translation_matrix @ H,  # 组合平移矩阵和单应矩阵
    (x_max - x_min, y_max - y_min)  # 输出图像尺寸
)

# 将图像复制到拼接结果的对应位置
fus_img[-y_min : h_a - y_min, -x_min : w_a - x_min] = img_a

# 保存结果图像
# 保存前两张图片的合并图
fig1 = plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB))
plt.title("Image A")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB))
plt.title("Image B")
plt.axis('off')

plt.tight_layout()
plt.savefig(f'{result_dir}/input_images.jpg', dpi=300, bbox_inches='tight')
plt.show()

# 保存匹配关键点图
fig2 = plt.figure(figsize=(10, 8))
plt.imshow(cv2.cvtColor(matched_keypoints_img, cv2.COLOR_BGR2RGB))
plt.title("Matched Keypoints")
plt.axis('off')
plt.tight_layout()
plt.savefig(f'{result_dir}/matched_keypoints.jpg', dpi=300, bbox_inches='tight')
plt.show()

# 保存融合结果图
fig3 = plt.figure(figsize=(10, 8))
plt.imshow(cv2.cvtColor(fus_img, cv2.COLOR_BGR2RGB))
plt.title("Fused Image")
plt.axis('off')
plt.tight_layout()
plt.savefig(f'{result_dir}/fused_image.jpg', dpi=300, bbox_inches='tight')
plt.show()

# 同时使用OpenCV保存原始图像（不包含标题和坐标轴）
cv2.imwrite(f'{result_dir}/input_images_cv.jpg', np.hstack((img_a, img_b)))
cv2.imwrite(f'{result_dir}/matched_keypoints_cv.jpg', matched_keypoints_img)
cv2.imwrite(f'{result_dir}/fused_image_cv.jpg', fus_img)

print("所有图像已保存到result文件夹中！")
print("保存的文件包括：")
print("- input_images.jpg (带标题的前两张图片)")
print("- matched_keypoints.jpg (匹配关键点图)")
print("- fused_image.jpg (融合结果图)")
print("- input_images_cv.jpg (OpenCV保存的前两张图片拼接)")
print("- matched_keypoints_cv.jpg (OpenCV保存的匹配关键点)")
print("- fused_image_cv.jpg (OpenCV保存的融合结果)")
