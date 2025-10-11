# =====================导入依赖库=====================
import cv2  # OpenCV库，用于图像处理
from skimage.util import random_noise  # scikit-image库的噪声添加功能
import numpy as np  # 数值计算库
from matplotlib import pyplot as plt  # 绘图库

# ====================读取原始图像=====================
img = cv2.imread('p1.jpg')  # 读取图像文件，OpenCV默认以BGR格式读取
# 获取图像中【100，100】这个像素的rbg三色
(b, g, r) = img[100, 100]  # 注意：OpenCV读取的图像是BGR顺序，不是RGB
# 打印这个像素点的rbg参数
print(b, g, r)  # 输出该像素点的蓝、绿、红通道值
# 输出原始图像
plt.imshow(img)  # 显示图像，但由于是BGR格式，颜色会失真
plt.title('Original Image')  # 设置图像标题
plt.savefig('result/original_bgr.jpg', dpi=300)  # 保存图像到result文件夹，分辨率为300dpi
plt.show()

# 颜色空间转换，将原始图像BGR格式 转换成 RGB格式，蓝色和红色互换，因为把'B'和'R'通道互换了
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 将BGR转换为RGB格式，恢复真实颜色
plt.imshow(rgb_img)  # 显示转换后的RGB图像
plt.title('RGB Image')  
plt.savefig('result/rgb_image.jpg', dpi=300)  # 保存RGB图像
plt.show()

# 将原始图像的RBG格式转换为灰度图
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 将BGR图像转换为灰度图像
plt.imshow(gray_img, cmap='gray')  # 显示灰度图，使用灰度色彩映射
plt.title('Gray Image')  
plt.savefig('result/gray_image.jpg', dpi=300)  # 保存灰度图像
plt.show() 

# ==========================添加噪声=========================
# mode='s&p'代表椒盐噪声，s代表白色，p代表黑色，amount=0.4代表会有40%的像素会随机替换为白色或者黑色像素
sp_noise_img = random_noise(rgb_img, mode='s&p', amount=0.4)  # 添加椒盐噪声，40%的像素受影响
# mode='gaussian' 代表高斯噪声，会在每个像素上添加随机偏差，服从高斯分布（均值=0，方差默认=0.01），我这里调整了均值和方差值
gus_noise_img = random_noise(rgb_img, mode='gaussian', mean=0.2, var=0.03)  # 添加高斯噪声，均值为0.2，方差为0.03

# 原图
plt.subplot(1, 3, 1)  # 创建1行3列的子图，当前为第1个
plt.imshow(rgb_img, cmap='gray')  # 显示原始RGB图像
plt.title('Original Image')

# 椒盐噪声
plt.subplot(1, 3, 2)  # 切换到第2个子图
plt.imshow(sp_noise_img, cmap='gray')  # 显示添加椒盐噪声后的图像
plt.title('S&P Noise')

# 高斯噪声
plt.subplot(1, 3, 3)  # 切换到第3个子图
plt.imshow(gus_noise_img, cmap='gray')  # 显示添加高斯噪声后的图像
plt.title('Gus Noise')

plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
plt.savefig('result/noise_comparison.jpg', dpi=300)  # 保存噪声对比图
plt.show()  # 显示图像


# =========================图像滤波=============================
# 均值滤波 - 用邻域内像素的平均值替换中心像素
mean_sp = cv2.blur(sp_noise_img, (5, 5))  # 对椒盐噪声图像进行5x5均值滤波
mean_gus = cv2.blur(gus_noise_img, (5, 5))  # 对高斯噪声图像进行5x5均值滤波

# 中值滤波（需要 uint8）- 用邻域内像素的中值替换中心像素，对椒盐噪声效果好
mid_sp = cv2.medianBlur((sp_noise_img*255).astype(np.uint8), 5)  # 将图像从[0,1]转换到[0,255]并进行5x5中值滤波
mid_gus = cv2.medianBlur((gus_noise_img*255).astype(np.uint8), 5)  # 同上，对高斯噪声图像处理

# 高斯滤波 - 使用高斯核进行加权平均滤波
gauss_sp = cv2.GaussianBlur((sp_noise_img*255).astype(np.uint8), (5, 5), 0)  # 5x5高斯滤波，标准差自动计算
gauss_gus = cv2.GaussianBlur((gus_noise_img*255).astype(np.uint8), (5, 5), 0)  # 同上，对高斯噪声图像处理

# =========================图像显示=============================
plt.figure(figsize=(13, 9))  # 创建13x9英寸的大图像窗口

# ---------- 第1行：椒盐噪声 ----------
plt.subplot(2, 3, 1)  # 第1个位置
plt.imshow(mean_sp)  # 显示均值滤波后的椒盐噪声图像
plt.title("S&P noise with Mean Filter")

plt.subplot(2, 3, 2)  # 第2个位置
plt.imshow(mid_sp)  # 显示中值滤波后的椒盐噪声图像
plt.title("S&P Noise with Median Filter")

plt.subplot(2, 3, 3)  # 第3个位置
plt.imshow(gauss_sp)  # 显示高斯滤波后的椒盐噪声图像
plt.title("S&P Noise with Gaussian Filter")

# ---------- 第2行：高斯噪声 ----------
plt.subplot(2, 3, 4)  # 第4个位置
plt.imshow(mean_gus)  # 显示均值滤波后的高斯噪声图像
plt.title("Gaussian noise with Mean Filter")

plt.subplot(2, 3, 5)  # 第5个位置
plt.imshow(mid_gus)  # 显示中值滤波后的高斯噪声图像
plt.title("Gaussian noise with Median Filter")

plt.subplot(2, 3, 6)  # 第6个位置
plt.imshow(gauss_gus)  # 显示高斯滤波后的高斯噪声图像
plt.title("Gaussian noise with Gaussian Filter")

plt.tight_layout()  # 自动调整子图间距
plt.savefig('result/filter_results_2x3.jpg', dpi=300)  # 保存滤波结果对比图
plt.show()


# ===========================手动实现中值滤波============================
def manual_median_filter_color(image, kernel_size=5):
    """
    手动实现彩色图像的中值滤波
    :param image: 输入彩色图像，numpy数组，shape=(H, W, 3)
    :param kernel_size: 滤波窗口大小
    :return: 中值滤波后的彩色图像
    """
    pad = kernel_size // 2  # 计算填充大小，确保边界也能处理
    # 对每个通道单独处理
    filtered_img = np.zeros_like(image)  # 创建与输入图像相同形状的零数组

    for c in range(3):  # 遍历通道 R,G,B
        channel = image[:, :, c]  # 提取当前通道
        padded_channel = np.pad(channel, pad_width=pad, mode='edge')  # 对通道进行边缘填充
        for i in range(channel.shape[0]):  # 遍历图像高度
            for j in range(channel.shape[1]):  # 遍历图像宽度
                region = padded_channel[i:i + kernel_size, j:j + kernel_size]  # 提取当前窗口区域
                filtered_img[i, j, c] = np.median(region)  # 计算区域中值并赋值

    return filtered_img  # 返回滤波后的图像


# 对之前添加椒盐噪声的图像进行手动中值滤波
manual_mid = manual_median_filter_color((sp_noise_img * 255).astype(np.uint8), kernel_size=5)  # 调用手动实现的中值滤波函数

# 显示对比
plt.figure(figsize=(8, 4))  # 创建8x4英寸的图像窗口

plt.subplot(1, 2, 1)
plt.imshow(sp_noise_img, cmap='gray')  # 显示原始椒盐噪声图像
plt.title("s&p noise img")

plt.subplot(1, 2, 2)
plt.imshow(manual_mid, cmap='gray')  # 显示手动中值滤波后的图像
plt.title("manual median filter") 

plt.tight_layout()
plt.savefig('result/manual_median.jpg', dpi=300)  # 保存手动中值滤波结果
plt.show()
