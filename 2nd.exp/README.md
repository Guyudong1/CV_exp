# 实验二：图像增强
## 一、实验目的
学会OpenCV的基本使用方法，利用OpenCV等计算机库对图像进行平滑、滤波等操作，实现图像增强。

## 二、实验内容

### 2.1 导入图像滤波相关的依赖包
- 1.OpenCV库：计算机视觉与图像处理 <br>
- 2.scikit-image库：图像增强与分析，random_noise 用于给图像添加噪声<br>
- 3.NumPy库：科学计算的基础库<br>
- 4.Matplotlib库：绘图与数据可视化<br>
```
# =====================导入依赖库=====================
import cv2
from skimage.util import random_noise
import numpy as np
from matplotlib import pyplot as plt
```

### 2.2 读取原始图像并进行色彩空间转换
- 读取计算机本地图像文件，获取并输入【100， 100】处像素点的RBG参数并输出
```
# ====================读取原始图像=====================
img = cv2.imread('p1.jpg')
# 获取图像中【100，100】这个像素的rbg三色
(b, g, r) = img[100, 100]
# 打印这个像素点的rbg参数
print(b, g, r)
# 输出原始图像
plt.imshow(img)
plt.title('Original Image')
plt.savefig('output_images/original_bgr.jpg', dpi=300)
plt.show()
```
```
- 输出：
        82 86 241
```
原图：<br>
<img src="https://github.com/user-attachments/assets/7b6c948b-e602-45d8-bdf2-7c470ab383dd" alt="rgb_image" width="400">

- 颜色空间转换，将原始图像BGR格式 转换成 RGB格式，蓝色和红色互换，因为把'B'和'R'通道互换了
```  
# 颜色空间转换，将原始图像BGR格式 转换成 RGB格式，蓝色和红色互换，因为把'B'和'R'通道互换了
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(rgb_img)
plt.title('RGB Image')
plt.savefig('output_images/rgb_image.jpg', dpi=300)
plt.show()
```
<img src="https://github.com/user-attachments/assets/d8b2b677-99d6-4fac-86a7-41b1fd27cade" width="400">

- 将原始图像的RBG格式转换为灰度图
```
# 将原始图像的RBG格式转换为灰度图
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(gray_img, cmap='gray')
plt.title('Gray Image')
plt.savefig('output_images/gray_image.jpg', dpi=300)
plt.show()
```
<img src="https://github.com/user-attachments/assets/10877d06-b003-4b89-b61f-86cac9add7f9" width="400">

### 2.3 添加噪声
```
# ==========================添加噪声=========================
# mode='s&p'代表椒盐噪声，s代表白色，p代表黑色，amount=0.4代表会有40%的像素会随机替换为白色或者黑色像素
sp_noise_img = random_noise(rgb_img, mode='s&p', amount=0.4)
# mode='gaussian' 代表高斯噪声，会在每个像素上添加随机偏差，服从高斯分布（均值=0，方差默认=0.01），我这里调整了均值和方差值
gus_noise_img = random_noise(rgb_img, mode='gaussian', mean=0.2, var=0.03)

# 原图
plt.subplot(1, 3, 1)
plt.imshow(rgb_img, cmap='gray')
plt.title('Original Image')

# 椒盐噪声
plt.subplot(1, 3, 2)
plt.imshow(sp_noise_img, cmap='gray')
plt.title('S&P Noise')

# 高斯噪声
plt.subplot(1, 3, 3)
plt.imshow(gus_noise_img, cmap='gray')
plt.title('Gus Noise')

plt.tight_layout()
plt.savefig('output_images/noise_comparison.jpg', dpi=300)
plt.show()
```
### 2.4 图像滤波
```
# =========================图像滤波=============================
# 均值滤波，用（5,5）的滤波核对椒盐噪声的去噪
mean_3 = cv2.blur(sp_noise_img, (5, 5))
# 中值滤波，用（5,5）的邻域，用邻域内的中位数替代中心像素度对椒盐噪声的去噪
mid_3 = cv2.medianBlur((sp_noise_img*255).astype(np.uint8), 5)
# 均值滤波，用（5,5）的滤波核对高斯噪声的去噪
gus_3 = cv2.blur(gus_noise_img, (5, 5))
# =========================图像显示=============================
plt.figure(figsize=(12, 4))  # 让图像显示更大更清晰

plt.subplot(1, 4, 1)
plt.imshow(sp_noise_img, cmap='gray')
plt.title("img with s&p noise")

plt.subplot(1, 4, 2)
plt.imshow(mean_3, cmap='gray')
plt.title("s&p noise img with mean")

plt.subplot(1, 4, 3)
plt.imshow(mid_3, cmap='gray')
plt.title("s&p noise img with median")

plt.subplot(1, 4, 4)
plt.imshow(gus_3, cmap='gray')
plt.title("gus noise img with mean")

plt.tight_layout()
plt.savefig('output_images/filter_results.jpg', dpi=300)
plt.show()
```

### ***2.5 手动实现一个滤波方式（中值滤波）

## 三、实验结果与分析

## 四、实验小结

