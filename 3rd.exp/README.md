# 实验三：图像关键点检测及图像拼接
#### 202310310169-顾禹东
## 一、实验目的
图像关键点检测及图像拼接实验报告，基于SIFT特征检测算法和FLANN特征匹配方法，实现两幅图像的特征点提取、匹配和自动拼接，掌握图像配准和融合的基本原理与实现方法。
## 二、实验内容
### 1.导入必要库
```
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
```
### 2.读取图像并预处理
```
# 读取图像
img_a = cv2.imread("img/a.jpg")
img_b = cv2.imread("img/b.jpg")
# 将图像转换为灰度图
gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
```
结果图：<br>
<img src="https://github.com/user-attachments/assets/0e34e5e0-6a9d-4eda-b93b-b4108c974aae" alt="rgb_image" width="400">

### 3.关键点检测
先使用FLANN算法快速找到两幅图像特征点之间的对应关系，然后再过滤掉不可靠的匹配点，只保留高质量的匹配，最后得到高质量的关键点
```
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

# 输出匹配统计信息
print("=== 特征匹配统计 ===")
print(f"初始匹配点对数: {len(matches)}")
print(f"筛选后优质匹配点对数: {len(good_matches)}")
print(f"过滤掉的匹配点对数: {len(matches) - len(good_matches)}")
print(f"优质匹配点比例: {len(good_matches)/len(matches):.2%}")
```
```
输出：
=== 特征匹配统计 ===
初始匹配点对数: 1063
筛选后优质匹配点对数: 295
过滤掉的匹配点对数: 768
优质匹配点比例: 27.75%
```
绘制对应关键点匹配情况
```
# 绘制匹配的SIFT关键点
matched_keypoints_img = cv2.drawMatches(
    img_a, kp_a, img_b, kp_b, good_matches,
    None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)
```
结果图：<br>
<img src="https://github.com/user-attachments/assets/7a9e3e53-0455-477d-b490-d01734958ee0" alt="rgb_image" width="430">
<img src="https://github.com/user-attachments/assets/9f0b5858-4c3b-464e-9b16-2f53591fc77c" alt="rgb_image" width="400">

### 4.图像拼接

```
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
```
结果图：<br>
<img src="https://github.com/user-attachments/assets/4ad09284-fe7f-45c6-962b-717f0f14a545" alt="rgb_image" width="400">

### 5、不规则图像拼接

## 三、实验结果与分析

## 四、实验总结与心得













