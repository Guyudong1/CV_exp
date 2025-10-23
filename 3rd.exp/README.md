# 实验三：图像关键点检测及图像拼接
#### 202310310169-顾禹东
## 一、实验目的
图像关键点检测及图像拼接实验报告，基于SIFT特征检测算法和FLANN特征匹配方法，实现两幅图像的特征点提取、匹配和自动拼接，掌握图像配准和融合的基本原理与实现方法。
## 二、实验内容
### 1.导入必要库
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
```
### 2.读取图像并预处理
```python
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
```python
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
```python
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

```python
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

这里的两张图像的拼接图出现黑边是因为透视变换造成的空白区域，将小图片扩大以匹配透视关系
<img src="https://github.com/user-attachments/assets/ef095c0b-a75b-448c-90bb-1d63f7b95fb7" alt="rgb_image" width="500">
<img src="https://github.com/user-attachments/assets/e4ea5f86-12f1-4ea6-83d3-13a42feb0b96" alt="rgb_image" width="400">

## 三、实验结果与分析

### 1. 特征检测与匹配效果分析
```
=== 特征匹配统计 ===
初始匹配点对数: 1063
筛选后优质匹配点对数: 295
过滤掉的匹配点对数: 768
优质匹配点比例: 27.75%
```
从统计结果可以看出，SIFT特征检测器在两幅图像中检测到了大量的特征点（1063对初始匹配），这体现了SIFT算法在特征检测方面的强大能力。然而，通过Lowe's比率测试后，仅有27.75%的匹配点被保留为优质匹配，这一结果表明：<br>
- 特征区分度：大部分特征点（72.25%）由于缺乏独特性而被过滤，说明图像中存在大量相似纹理区域，这些区域容易产生误匹配。<br>
- 匹配质量：27.75%的优质匹配比例在实际应用中属于可接受范围，表明算法能够有效识别出具有区分度的稳定特征点。<br>
- 阈值选择：使用0.7的比率阈值是一个经验值，在实际应用中可以根据具体图像特点进行调整。降低阈值可以提高匹配精度但会减少匹配数量，增加阈值则相反。<br>

### 2. 图像拼接质量评估
- 成功方面：<br>
  - 几何对齐精度：从拼接结果可以看出，两幅图像在重叠区域实现了较好的几何对齐，建筑物轮廓、窗户等结构特征基本吻合，说明单应性矩阵估计准确。
  - 特征点分布：优质匹配点在图像中分布相对均匀，既有角点特征也有边缘特征，这为稳定的变换估计提供了基础。
  - 变换连续性：透视变换保持了图像的几何连续性，没有出现明显的断裂或扭曲现象。
- 存在问题：<br>
  - 亮度差异：两幅图像在拼接处存在轻微的亮度差异，这是由于拍摄时光照条件不同导致的。
  - 细节配准误差：在放大观察时，可以发现某些细节区域存在微小的配准误差，这可能是由于特征点定位精度限制或变换模型简化所致。
  - 黑边现象：在部分拼接结果中出现黑色边框，这是透视变换过程中产生的空白区域。

### 3. 影响因素分析
- 图像质量因素：
  - 图像分辨率和清晰度影响特征检测数量和质量
  - 光照条件影响特征描述符的稳定性
  - 纹理丰富程度决定可检测特征点的数量

- 算法参数影响：
  - SIFT特征点数量阈值影响检测灵敏度
  - Lowe's比率阈值影响匹配精度和数量平衡
  - RANSAC重投影误差阈值影响内点筛选严格程度
## 四、实验总结与心得













