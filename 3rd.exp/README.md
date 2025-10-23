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
![color_vs_grayscale](https://github.com/user-attachments/assets/0e34e5e0-6a9d-4eda-b93b-b4108c974aae)

## 三、实验结果与分析

## 四、实验总结与心得













