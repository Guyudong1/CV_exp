# =====================导入依赖库=====================
import cv2
from skimage.util import random_noise
import numpy as np
from matplotlib import pyplot as plt

# ====================读取原始图像=====================
img = cv2.imread('p1.jpg')
# 获取图像中【100，100】这个像素的rbg三色
(b, g, r) = img[100, 100]
# 打印这个像素点的rbg参数
print(b, g, r)
# 输出原始图像
plt.imshow(img)
plt.title('Original Image')
plt.savefig('result/original_bgr.jpg', dpi=300)
plt.show()

# 颜色空间转换，将原始图像BGR格式 转换成 RGB格式，蓝色和红色互换，因为把'B'和'R'通道互换了
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(rgb_img)
plt.title('RGB Image')
plt.savefig('result/rgb_image.jpg', dpi=300)
plt.show()

# 将原始图像的RBG格式转换为灰度图
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(gray_img, cmap='gray')
plt.title('Gray Image')
plt.savefig('result/gray_image.jpg', dpi=300)
plt.show()

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
plt.savefig('result/noise_comparison.jpg', dpi=300)
plt.show()


# =========================图像滤波=============================
# 均值滤波，用（5,5）的滤波核对椒盐噪声的去噪
mean_3 = cv2.blur(sp_noise_img, (5, 5))
# 中值滤波，用（5,5）的邻域，用邻域内的中位数替代中心像素度对椒盐噪声的去噪
mid_3 = cv2.medianBlur((sp_noise_img*255).astype(np.uint8), 5)
# 均值滤波，用（5,5）的滤波核对高斯噪声的去噪
gus_3 = cv2.blur(gus_noise_img, (5, 5))
# 图像显示
plt.figure(figsize=(12, 4))

# 椒盐噪声图
plt.subplot(1, 4, 1)
plt.imshow(sp_noise_img, cmap='gray')
plt.title("img with s&p noise")
# 用均值滤波去噪后的椒盐噪声图
plt.subplot(1, 4, 2)
plt.imshow(mean_3, cmap='gray')
plt.title("s&p noise img with mean")
# 用中值滤波去噪后的椒盐噪声图
plt.subplot(1, 4, 3)
plt.imshow(mid_3, cmap='gray')
plt.title("s&p noise img with median")
# 用均值滤波去噪后的高斯噪声图
plt.subplot(1, 4, 4)
plt.imshow(gus_3, cmap='gray')
plt.title("gus noise img with mean")

plt.tight_layout()
plt.savefig('result/filter_results.jpg', dpi=300)
plt.show()
