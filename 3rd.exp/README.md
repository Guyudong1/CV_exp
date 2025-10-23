# 实验三：图像关键点检测及图像拼接
#### 202310310169-顾禹东

> **文件说明**  
> 本实验的代码在`code.py`文件中，
> 本实验的输入图像在`img`文件夹中，
> 本实验的输出结果在`result`文件夹中。

## 一、实验目的
图像关键点检测及图像拼接实验报告，基于SIFT特征检测算法和FLANN特征匹配方法，实现两幅图像的特征点提取、匹配和自动拼接，掌握图像配准和融合的基本原理与实现方法。
## 二、实验内容
### 1.导入必要库
- cv2:OpenCV库 <br>
- numpy:Numpy库 <br>
- matplotlib.pyplot:可视化库 <br>
- os:操作系统交互的核心库 <br>
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
```
### 2.读取图像并预处理
这里先读取两张图像作为拼接的原素材， 再将图像转换为灰度图，
原因：
- 效率：大幅减少计算量
- 鲁棒性：提高对光照和颜色变化的稳定性
- 专注性：聚焦于结构纹理信息，避免颜色干扰
- 标准化：符合大多数经典特征检测算法的输入要求
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
#### 图像拼接的几何变换核心流程
- 图像拼接的几何变换过程始于特征点坐标的精确提取，其中`src_pts`从待变换的源图像（img_b）中获取匹配关键点坐标，而`dst_pts`则从基准图像（img_a）中提取对应点坐标，这些点对建立了两幅图像之间的空间对应关系。接着，系统采用RANSAC算法鲁棒地估计3×3单应性矩阵H，该矩阵定义了从img_b到img_a的透视变换关系，其中的重投影误差阈值（5.0）有效控制了内点筛选的严格程度，确保在存在部分错误匹配的情况下仍能获得准确的变换参数。<br>
- 在确定几何关系后，算法计算img_b四个角点经过透视变换后的新位置，通过将变换后的img_b角点与原始img_a角点合并，计算出所有角点的最小和最大坐标值，从而确定最终拼接图像的完整尺寸范围。为了处理变换后可能出现的负坐标问题，系统创建了一个平移矩阵，将整个变换结果平移到正坐标区域，确保所有像素都位于可视范围内。<br>
- 最后，通过对img_b应用组合变换（透视变换与平移变换的结合），并将img_a精确复制到变换后图像的对应位置，实现了两幅图像的无缝空间对齐和内容融合。这一完整的"特征匹配→几何估计→图像变换"技术流水线，体现了计算机视觉中图像配准的核心原理，为生成高质量的全景拼接图像奠定了坚实基础。整个过程中，单应性矩阵发挥了关键的坐标映射作用，而RANSAC算法则保证了在复杂场景下的算法鲁棒性，使得即使存在一定比例的误匹配，仍能获得令人满意的拼接效果。<br>
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
- 1.尺寸不匹配：<br>
这里的两张图像的拼接图出现黑边是因为透视变换造成的空白区域，将小图片扩大以匹配透视关系。
<img src="https://github.com/user-attachments/assets/ef095c0b-a75b-448c-90bb-1d63f7b95fb7" alt="rgb_image" width="500">
<img src="https://github.com/user-attachments/assets/e4ea5f86-12f1-4ea6-83d3-13a42feb0b96" alt="rgb_image" width="400"> <br>

- 2.亮度条件：<br>
这里用亮度对比度的调整模拟不同光照条件下的拼接情况，可以看到因为光照条件的不同情况下关键点的匹配会出现变化导致图片拼接时出现细微误差。
<img src="https://github.com/user-attachments/assets/fa8fab23-d83e-4ded-a5ba-d1111e9f6fe8" alt="rgb_image" width="500">
<img src="https://github.com/user-attachments/assets/d11d0350-ee12-44e2-b6da-88cd8b3ffdce" alt="rgb_image" width="400">

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
### 1. 技术收获
- 特征检测方面：<br>
通过本次实验，我深入理解了SIFT特征检测算法的工作原理。SIFT通过构建高斯金字塔和DoG空间来检测尺度不变的特征点，然后基于局部梯度方向生成具有旋转不变性的描述符。这种方法的优势在于其对尺度、旋转和光照变化的鲁棒性，使其成为图像匹配和拼接任务中的首选算法。
- 特征匹配方面：<br>
实验让我掌握了FLANN匹配器的使用方法和参数调优技巧。FLANN通过构建KD树索引实现高效的近似最近邻搜索，在大规模特征匹配中显著提高了效率。同时，Lowe's比率测试的应用让我认识到特征匹配质量评估的重要性，通过距离比率筛选可以有效提高匹配的可靠性。
- 图像配准方面：<br>
单应性矩阵的估计是图像拼接的核心环节。通过RANSAC算法估计透视变换矩阵，我理解了如何从含有噪声的匹配点集中鲁棒地估计变换参数。这个过程不仅涉及线性代数中的矩阵运算，还包括对投影几何的理解。

### 2. 实践体会
- 参数调优的重要性：
在实验过程中，我发现算法参数的选择对最终结果有显著影响。例如：
  - FLANN匹配器中trees参数影响搜索精度，checks参数影响搜索深度
  - Lowe's比率阈值需要在匹配数量和匹配质量之间权衡
  - RANSAC的重投影误差阈值直接影响内点的筛选标准
这些参数需要根据具体图像特点和任务需求进行精心调整。
- 问题诊断能力：
通过观察中间结果和统计分析，我学会了如何诊断拼接过程中出现的问题。比如当匹配点数量过少时，可能是特征检测参数不合适；当拼接出现明显错位时，可能是RANSAC阈值设置不当或存在大量错误匹配。









