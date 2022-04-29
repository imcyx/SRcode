# 基于SRGAN的图像超分辨率处理

# 相关代码

论文地址： [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://openaccess.thecvf.com/content_cvpr_2017/papers/Ledig_Photo-Realistic_Single_Image_CVPR_2017_paper.pdf)

复现代码： [Github-SRcode](https://github.com/imcyx/SRcode)

# 算法原理

## 图像超分任务

图像分辨率是一组用于评估图像中蕴含细节信息丰富程度的性能参数，包括时间分辨率、空间分辨率及色阶分辨率等，体现了成像系统实际所能反映物体细节信息的能力。相较于低分辨率图像，高分辨率图像通常包含更大的像素密度、更丰富的纹理细节及更高的可信赖度。但在实际上情况中，受采集设备与环境、网络传输介质与带宽、图像退化模型本身等诸多因素的约束，我们通常并不能直接得到具有边缘锐化、无成块模糊的理想高分辨率图像。提升图像分辨率的最直接的做法是对采集系统中的光学硬件进行改进，但是由于制造工艺难以大幅改进并且制造成本十分高昂，因此物理上解决图像低分辨率问题往往代价太大。由此，从软件和算法的角度着手，实现图像超分辨率重建的技术成为了图像处理和计算机视觉等多个领域的热点研究课题。

图像的超分辨率重建技术指的是将给定的低分辨率图像通过特定的算法恢复成相应的高分辨率图像。具体来说，图像超分辨率重建技术指的是利用数字图像处理、计算机视觉等领域的相关知识，借由特定的算法和处理流程，从给定的低分辨率图像中重建出高分辨率图像的过程。其旨在克服或补偿由于图像采集系统或采集环境本身的限制，导致的成像图像模糊、质量低下、感兴趣区域不显著等问题。

简单来理解超分辨率重建就是将小尺寸图像变为大尺寸图像，使图像更加“清晰”。具体效果如下图所示。

![img](https://gitee.com/CYX12138/cloudimage/raw/master/img_2022_04/202204291801599.jpeg)

现今，超分辨率问题的病态性质尤其表现在取较高的放大因子时，重构的超分辨率图像通常会缺失纹理细节。监督SR算法的优化目标函数通常取重建高分辨率图像和地面真值之间的均方误差，在减小均方误差的同时又可以增大峰值信噪比(PSNR)，PSNR是评价和比较SR算法的常用指标。但是MSE和PSNR值的高低并不能很好的表示视觉效果的好坏。正如Figture2表现出的，PSNR最高并不能反映最好的视觉SR效果。

SRGAN，由论文《[Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://openaccess.thecvf.com/content_cvpr_2017/papers/Ledig_Photo-Realistic_Single_Image_CVPR_2017_paper.pdf)》提出。文章使用了结合跳跃-连接（skip-connection）的深度残差网络（ResNet）。通过使用VGG网络的高层特征映射定义了新的感知损失，该损失使用的判别器使生成的高分辨率图像与实际原始图像在视觉上尽量相似。

本文实验原理基于SRGAN的图像超分方法，下面分析该论文的理论内容：

## 文章创新点

1. 构建了基于`MSE`损失构建的16 blocks ResNet：`SRResNet`，作为生成网络的backbone 。
2. 提出了基于感知损失优化的`SRGAN`网络，同时将内容损失由直接`MSE`替换为VGG网络特征图的欧氏距离计算损失。引入的判别器结构增加了生成图片的真实感，改进的感知损失使与原始图片的相似不再局限于像素而是全局。
3. 使用主观评估手段：`MOS`，更加强调人的感知。

## 网络结构

核心目标：训练一个生成器 $G$，对低分辨率图像进行超分恢复。

### 生成网络：

![image-20220411201742891](https://gitee.com/CYX12138/cloudimage/raw/master/img/202204112017588.png)



### 判别网络：

![image-20220411201903779](https://gitee.com/CYX12138/cloudimage/raw/master/img/202204112019838.png)

![判别网络](https://gitee.com/CYX12138/cloudimage/raw/master/img_2022_04/202204291808748.png)

## 损失函数

$ I^{HR} $：原始高分辨率图片

$ I^{SR} $：超分辨率恢复照片

$ I^{LR} $：原始高分辨率图片`高斯滤波`+`bicubic 下采样`后的低分辨率图片

生成器参数更新：
$$
\hat{\theta}_{G}=\arg \min _{\theta_{G}} \frac{1}{N} \sum_{n=1}^{N} l^{S R}\left(G_{\theta_{G}}\left(I_{n}^{L R}\right), I_{n}^{H R}\right)
$$
这段公式是生成网络优化的核心：由原始高分图像下采样的低分图像经生成器恢复后，与原始高分图像计算损失，再对总损失沿负梯度方向优化。

而作为文章的创新之一，损失函数 $l^{SR}$ 作者并没有用通常的MSE loss，而是提出基于改进的感知损失函数(Perceptual loss function) ，它由content loss 和 adversarial loss 加权获得：
$$
l^{S R}=\underbrace{\underbrace{l_{\mathrm{X}}^{S R}}_{\text {content loss }}+\underbrace{10^{-3} l_{G e n}^{S R}}_{\text {adversarial loss}}}_{\text {perceptual loss (for VGG based content losses)}}
$$
### content loss

通常的 content loss 由逐像素的 MSE loss 表示：
$$
l_{M S E}^{S R}=\frac{1}{r^{2} W H} \sum_{x=1}^{r W} \sum_{y=1}^{r H}\left(I_{x, y}^{H R}-G_{\theta_{G}}\left(I^{L R}\right)_{x, y}\right)^{2}
$$
而在这里，作者取代传统的逐像素 MSE loss，使用 VGG loss，更加考虑全局相关性 。这里的 $\phi_{i, j}$ 表示在 VGG19 网络中的第 i 个最大池化层之前通过第 j 个卷积（激活后）获得的 $W \times H$ 特征图，使用原图和生成图像经过VGG19后特征图的欧氏距离表示 loss ，计算表达式如下：
$$
\begin{aligned}
l_{V G G / i . j}^{S R}=\frac{1}{W_{i, j} H_{i, j}} & \sum_{x=1}^{W_{i, j}} \sum_{y=1}^{H_{i, j}}\left(\phi_{i, j}\left(I^{H R}\right)_{x, y}\right.\left.-\phi_{i, j}\left(G_{\theta_{G}}\left(I^{L R}\right)\right)_{x, y}\right)^{2}
\end{aligned}
$$
### adversarial loss

对于 adversarial loss $l_{G e n}^{S R}$ ，作者基于判别器在所有训练样本上的判别准确率之和，$ D_{\theta_{D}}\left(G_{\theta_{G}}\left(I^{L R}\right)\right) $ 表示的是判别器判断生成图像为原始高分图像的概率。为了更好的梯度下降效果，使用 $ -\log D_{\theta_{D}}\left(G_{\theta_{G}}\left(I^{L R}\right)\right) $ 代替 $\log \left[ {1 - {D_{{\theta _D}}}\left( {{G_{{\theta _G}}}\left( {{I^{LR}}} \right)} \right)} \right]$。
$$
l_{G e n}^{S R}=\sum_{n=1}^{N}-\log D_{\theta_{D}}\left(G_{\theta_{G}}\left(I^{L R}\right)\right)
$$

## 评价标准

作者提出了一种新的评价标准：平均意见分数(Mean opinion score)。

具体来说，要求 26 位评分者为超分图像`SR`分配从 1（质量差）到 5（质量好）的积分。每个评分者对数据集每张图片的12个处理结果进行评判，包括：最近邻(NN)、bicubic、SRCNN、 SelfExSR 、DRCN 、 ESPCN、 SRResNet-MSE、SRResNet-VGG22∗ （∗ 表示不在 BSD100 数据集评分）、SRGAN-MSE∗、SRGAN-VGG22∗、 SRGAN-VGG54 和原始`HR`图像。评分者同时在来自 BSD300 训练集的 20 张图像的 NN（得分 1）和 HR（得分5）上进行了校准。

经过实验，改评价标准具有良好的可靠性，相同图像的评级之间没有显着差异。 评分者非常一致地将 NN 插值测试图像评为 1，将原始 HR 图像评为 5（参见下图）。

![image-20220417214452192](https://gitee.com/CYX12138/cloudimage/raw/master/img/202204172144239.png)

## 实验设置

1. 使用`ImageNet`数据集作为数据来源，随机从中选取图像作为训练，并将其与测试图像区分开。
2. 每个mini-batch，训练的16张`HR`图像是对原图进行随机96X96裁剪获得的。
3. 对`HR`图像再使用bicubic内核进行4X下采样，获得24X24大小的`LR`图像。
4. `LR`图像标准化到 $/255 \in \left[ {0,1} \right]$ 中，`HR`图像标准化到$/127.5 - 1 \in \left[ { - 1,1} \right]$中。
5. `SRResNet`训练使用Adam优化器，β取0.9，learning rate 取 1e-4，进行1e6次迭代。为了避免局部最优，先对`SRenNet`进行训练，并将训练结果作为生成器的初始权重。
6. `SRGAN`的训练也使用`Adam`优化器，先以 1e-4 的 learning rate 进行 1e5 次迭代，再以 1e-5 的 learning rate 进行 1e5 次迭代。每轮生成器和判别器参数交替更新。

|           | SRResNet-MSE | SRResNet-VGG22 | SRGAN-MSE | SRGAN-VGG22 | SRGAN-VGG54 |
| :-------: | :----------: | :------------: | :-------: | :---------: | :---------: |
| **Set5**  |              |                |           |             |             |
|   PSNR    |    32.05     |     30.51      |   30.64   |    29.84    |    29.40    |
|   SSIM    |    0.9019    |     0.8803     |  0.8701   |   0.8468    |   0.8472    |
|    MOS    |     3.37     |      3.46      |   3.77    |    3.78     |    3.58     |
| **Set14** |              |                |           |             |             |
|   PSNR    |    28.49     |     27.19      |   26.92   |    26.44    |    26.02    |
|   SSIM    |    0.8184    |     0.7807     |  0.7611   |   0.7518    |   0.7397    |
|    MOS    |     2.98     |     3.15*      |   3.43    |    3.57     |    3.72*    |

- SRGAN-VGG22: $l_{V G G / 2.2}^{S R}$ ，$\phi_{2,2}$ 表示较低级特征的特征图上定义的损失。
- SRGAN-VGG54: $l_{V G G / 5.4}^{S R}$， $\phi_{5,4}$ 表示更深网络层的更高级特征图上定义的损失，有更多的潜力去关注图像内容。

表格的前两列表示基于`SRResNet`的消融实验，分别用普通的`MSE`损失和特征损失。可以看出，即使使用了对抗网络，`MSE`损失相比感知损失也提供了更高的`PSNR`结果，然而在实际结果给人的感知上，图片却更加平滑且难以让人信服，但`MOS`的结果却真实反映出了这一结果。

同时，经过比较，使用感知损失相比MSE在`Set5`上的差别并不大，但在`Set14`上，SRGAN-VGG54 的`MOS`分数明显优于其它方法。而且对比$\phi_{2,2}$，使用更高级别特征图的$\phi_{5,4}$会产生更优秀的纹理细节。

# 代码复现

```python
"""
Super-resolution of CelebA using Generative Adversarial Networks.

The dataset can be downloaded from: https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADIKlz8PR9zr6Y20qbkunrba/Img/img_align_celeba.zip?dl=0

Instrustion on running the script:
1. Download the dataset from the provided link
2. Save the folder 'img_align_celeba' to 'datasets/'
4. Run the sript using command 'python srgan.py'
"""

from __future__ import print_function, division

import os
import datetime
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageFilter
from glob import glob
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D, Add
from tensorflow.keras.layers import PReLU, LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

matplotlib.use('Agg')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class DataLoader(object):
    def __init__(self, dataset_name, img_res=(256, 256)):
        self.dataset_name = dataset_name
        self.img_res = img_res
        self.image_paths = glob('./datasets/%s/*' % self.dataset_name)
        np.random.shuffle(self.image_paths)
        self.images = self.image_paths

    def reload_dataset(self):
        self.images = self.image_paths

    def load_data(self, batch_size=1, is_testing=False):
        # without replacement sampling
        batch_images = np.random.choice(self.images, size=1, replace=False)
        img = Image.open(batch_images[0]).convert('RGB')
        # if high_res image smaller than set size, continue select
        while img.size[0] - self.img_res[0] - 1 <= 0 or img.size[1] - self.img_res[1] - 1 <= 0:
            batch_images = np.random.choice(self.images, size=1, replace=False)
            img = Image.open(batch_images[0]).convert('RGB')

        imgs_hr = []
        imgs_lr = []
        for _ in range(batch_size):
            # random crop 96 × 96 HR sub images
            left_upx = np.random.randint(0, img.size[0] - self.img_res[0] - 1)
            left_upy = np.random.randint(0, img.size[1] - self.img_res[1] - 1)
            img_hr = img.crop((left_upx, left_upy, left_upx + self.img_res[0], left_upy + self.img_res[1]))
            # Gaussian filter
            img_lr = img_hr.filter(ImageFilter.GaussianBlur(1.5))
            # Downsampling
            img_lr = img_lr.resize((self.img_res[0] // 4, self.img_res[1] // 4), Image.Resampling.BICUBIC)
            # convert to numpy
            img_hr = np.array(img_hr)
            img_lr = np.array(img_lr)

            # If training => do random flip
            if not is_testing and np.random.random() < 0.5:
                img_hr = np.fliplr(img_hr)
                img_lr = np.fliplr(img_lr)

            imgs_hr.append(img_hr)
            imgs_lr.append(img_lr)

        imgs_hr = np.array(imgs_hr) / 127.5 - 1.
        imgs_lr = np.array(imgs_lr) / 127.5 - 1.

        return imgs_hr, imgs_lr

    def load_spec_data(self):
        os.makedirs(f'images/_src', exist_ok=True)
        path = glob('./datasets/img_test/*')

        imgs_hr = []
        imgs_lr = []
        for i, img_path in enumerate(path):
            img_hr = Image.open(img_path).convert('RGB')
            # Gaussian filter
            img_lr = img_hr.filter(ImageFilter.GaussianBlur(1.5))
            # Downsampling
            img_lr = img_lr.resize((img_hr.size[0] // 4, img_hr.size[1] // 4), Image.Resampling.BICUBIC)
            img_lr.save(f'./images/_src/{i}.png')
            # convert to numpy
            imgs_hr.append(np.array(img_hr))
            imgs_lr.append(np.array(img_lr))

        imgs_hr = np.array(imgs_hr) / 127.5 - 1.
        imgs_lr = np.array(imgs_lr) / 127.5 - 1.
        return imgs_hr, imgs_lr

class SRGAN():
    def __init__(self):
        # Input shape
        # Use conv layers totally. So do not need to choose shape.
        self.channels = 3
        self.lr_shape = (None, None, self.channels)
        self.hr_shape = (None, None, self.channels)

        # Number of residual blocks in the generator
        self.n_residual_blocks = 16

        # Number of filters in the first layer of G and D
        self.gf = self.df =64

        # Configure data loader
        # self.dataset_name = 'img_align_celeba'
        self.dataset_name = 'test2017'
        self.data_loader = DataLoader(dataset_name=self.dataset_name)

        self.test_imgs = self.data_loader.load_spec_data()

        optimizer = Adam(0.0001, 0.9)

        # We use a pre-trained VGG19 model to extract image features from the high resolution
        # and the generated high resolution images and minimize the mse between them
        vgg = VGG19(weights="imagenet", input_shape=self.hr_shape, include_top=False)
        self.vgg = Model(vgg.input, outputs=vgg.layers[9].output, trainable=False)
        self.vgg.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # High res. and low res. images
        img_hr = Input(self.hr_shape)
        img_lr = Input(self.lr_shape)

        # Build the generator
        self.generator = self.build_generator()
        self.generator.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
        # Generate high res. version from low res.
        fake_hr = self.generator(img_lr)

        # Perceptual loss: 1e-3 * adversarial loss + vgg loss
        self.combined = Model([img_lr, img_hr], [self.discriminator(fake_hr), self.vgg(fake_hr)])
        self.combined.compile(loss=['binary_crossentropy', 'mse'], loss_weights=[1e-3, 1/(12.75**2)], optimizer=optimizer)

        plot_model(
            self.generator,  # keras模型
            to_file= "生成网络.png",  # 保存图片路径
            show_shapes=True,  # 是否显示形状信息
            show_layer_names=True,  # 是否显示图层名称
            rankdir="TB",  # "TB":垂直图  "LR":水平图
            expand_nested=True,  # 是否将嵌套模型展开为簇。
            dpi=96  # 图片每英寸点数。
        )

        plot_model(
            self.discriminator,  # keras模型
            to_file= "判别网络.png",  # 保存图片路径
            show_shapes=True,  # 是否显示形状信息
            show_layer_names=True,  # 是否显示图层名称
            rankdir="TB",  # "TB":垂直图  "LR":水平图
            expand_nested=True,  # 是否将嵌套模型展开为簇。
            dpi=96  # 图片每英寸点数。
        )

    def build_generator(self):

        def residual_block(layer_input):
            """Residual block described in paper"""
            d = Conv2D(64, kernel_size=3, strides=1, padding='same')(layer_input)
            d = Activation('relu')(d)
            d = BatchNormalization(momentum=0.8)(d)
            d = Conv2D(64, kernel_size=3, strides=1, padding='same')(d)
            d = BatchNormalization(momentum=0.8)(d)
            d = Add()([d, layer_input])
            return d

        def deconv2d(layer_input):
            """Layers used during upsampling"""
            u = Conv2D(256, kernel_size=3, strides=1, padding='same')(layer_input)
            u = UpSampling2D(size=2)(u)
            u = Activation('relu')(u)
            return u

        # Low resolution image input
        img_lr = Input(shape=(None, None, 3))

        # Pre-residual block
        c1 = Conv2D(64, kernel_size=9, strides=1, padding='same')(img_lr)
        c1 = Activation('relu')(c1)

        # Propogate through residual blocks
        r = residual_block(c1)
        for _ in range(self.n_residual_blocks - 1):
            r = residual_block(r)

        # Post-residual block
        c2 = Conv2D(64, kernel_size=3, strides=1, padding='same')(r)
        c2 = BatchNormalization(momentum=0.8)(c2)
        c2 = Add()([c2, c1])

        # Upsampling
        u1 = deconv2d(c2)
        u2 = deconv2d(u1)

        # Generate high resolution output
        gen_hr = Conv2D(self.channels, kernel_size=9, strides=1, padding='same', activation='tanh')(u2)
        
        return Model(img_lr, gen_hr)

    def build_discriminator(self):

        def d_block(layer_input, filters, strides=1, bn=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=3, strides=strides, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        # Input img
        d0 = Input(shape=(None, None, 3))

        d1 = d_block(d0, self.df, bn=False)
        d2 = d_block(d1, self.df, strides=2)
        d3 = d_block(d2, self.df*2)
        d4 = d_block(d3, self.df*2, strides=2)
        d5 = d_block(d4, self.df*4)
        d6 = d_block(d5, self.df*4, strides=2)
        d7 = d_block(d6, self.df*8)
        d8 = d_block(d7, self.df*8, strides=2)

        d9 = Dense(self.df*16)(d8)
        d10 = LeakyReLU(alpha=0.2)(d9)
        validity = Dense(1, activation='sigmoid')(d10)

        return Model(d0, validity)

    def pre_train(self, epochs, batch_size=1, sample_interval=50):
        self.generator.load_weights('./weights/pre_training_checkpoints/')
        self.data_loader.reload_dataset()
        start_time = datetime.datetime.now()
        last_time = datetime.datetime.now()
        for epoch in range(epochs):
            # ------------------
            #  Train Generator
            # ------------------
            # Sample images and their conditioning counterparts
            imgs_hr, imgs_lr = self.data_loader.load_data(batch_size)
            # Train the generators
            self.generator.train_on_batch(imgs_lr, imgs_hr)

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.generator.save_weights('./weights/pre_training_checkpoints/')

                loss = self.evaluate(epoch, comp_dir="_res_gen")

                elapsed_time = datetime.datetime.now() - start_time
                used_time = datetime.datetime.now() - last_time
                # Plot the progress
                print(f"epoch: {epoch}\t g_loss: {loss}\t time: {elapsed_time}\t interval: {used_time}")
                last_time = datetime.datetime.now()

    def train(self, epochs, batch_size=1, sample_interval=50):
        self.generator.load_weights('./weights/pre_training_checkpoints/')
        self.data_loader.reload_dataset()
        start_time = datetime.datetime.now()
        last_time = datetime.datetime.now()
        # self.generator.summary()
        # self.discriminator.summary()
        for epoch in range(epochs):
            # ----------------------
            #  Train Discriminator
            # ----------------------

            # Sample images and their conditioning counterparts
            imgs_hr, imgs_lr = self.data_loader.load_data(batch_size)

            # From low res. image generate high res. version
            fake_hr = self.generator.predict(imgs_lr)

            # Calculate output shape of D (PatchGAN)
            patch_w = imgs_hr.shape[1] // 2 ** 4
            patch_h = imgs_hr.shape[2] // 2 ** 4
            valid = np.ones((batch_size, patch_w, patch_h, 1))
            fake = np.zeros((batch_size, patch_w, patch_h, 1))

            # Train the discriminators (original images = real / generated = Fake)
            d_loss_real = self.discriminator.train_on_batch(imgs_hr, valid)
            d_loss_fake = self.discriminator.train_on_batch(fake_hr, fake)
            d_loss = (0.5 * np.add(d_loss_real, d_loss_fake))[0]

            # ------------------
            #  Train Generator
            # ------------------

            # Sample images and their conditioning counterparts
            imgs_hr, imgs_lr = self.data_loader.load_data(batch_size)

            # The generators want the discriminators to label all the generated images as real
            patch_w = imgs_hr.shape[1] // 2 ** 4
            patch_h = imgs_hr.shape[2] // 2 ** 4
            valid = np.ones((batch_size, patch_w, patch_h, 1))

            # Extract ground truth image features using pre-trained VGG19 model
            image_features = self.vgg.predict(imgs_hr)

            # Train the generators
            self.combined.train_on_batch([imgs_lr, imgs_hr], [valid, image_features])

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.generator.save_weights('./weights/training_checkpoints/')
                # checkpoint = tf.train.Checkpoint(self.generator)
                # checkpoint.save('./weights/training_checkpoints')

                # self.sample_images(epoch)
                loss = self.evaluate(epoch, comp_dir="_res_gan", src_dir="_res_src_gan")

                elapsed_time = datetime.datetime.now() - start_time
                used_time = datetime.datetime.now() - last_time
                # Plot the progress
                print (f"epoch: {epoch}\t d_loss: %.5f\t g_loss: (%.5f, %.5f)\t time: {elapsed_time}\t interval: {used_time}"
                       %(d_loss, loss[0], loss[1]))
                last_time = datetime.datetime.now()


    def sample_images(self, epoch):
        os.makedirs('images/%s' % self.dataset_name, exist_ok=True)
        r, c = 2, 2

        imgs_hr, imgs_lr = self.data_loader.load_data(batch_size=2, is_testing=True)
        fake_hr = self.generator.predict(imgs_lr)

        # Rescale images 0 - 1
        imgs_lr = 0.5 * imgs_lr + 0.5
        fake_hr = 0.5 * fake_hr + 0.5
        imgs_hr = 0.5 * imgs_hr + 0.5

        # Save generated images and the high resolution originals
        titles = ['Generated', 'Original']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for row in range(r):
            for col, image in enumerate([fake_hr, imgs_hr]):
                axs[row, col].imshow(image[row])
                axs[row, col].set_title(titles[col])
                axs[row, col].axis('off')
            cnt += 1
        fig.savefig("images/%s/%d.png" % (self.dataset_name, epoch))
        plt.close()

        # # Save low resolution images for comparison
        # for i in range(r):
        #     fig = plt.figure()
        #     plt.imshow(imgs_lr[i])
        #     fig.savefig('images/%s/%d_lowres%d.png' % (self.dataset_name, epoch, i))
        #     plt.close()


    def evaluate(self, epoch, comp_dir='', src_dir='', testing=False):
        if testing:
            self.generator.load_weights('./weights/pre_training_checkpoints/')

        r, c = 2, 3
        imgs_hr = self.test_imgs[0]
        imgs_lr = self.test_imgs[1]
        fake_hr = self.generator.predict(imgs_lr)
        precision, vgg_feature = self.combined.predict([imgs_lr, imgs_hr])

        # MSE loss
        MSE_loss = MeanSquaredError()(fake_hr, imgs_hr).numpy()
        # Perceptual loss
        patch_w = imgs_hr.shape[1] // 2 ** 4
        patch_h = imgs_hr.shape[2] // 2 ** 4
        valid = np.ones((c, patch_w, patch_h, 1))
        content_loss = MeanSquaredError()(self.vgg.predict(imgs_hr), vgg_feature).numpy()
        adversarial_loss = BinaryCrossentropy()(valid, precision).numpy()
        Perceptual_loss = content_loss/(12.75**2) + adversarial_loss/1000

        # Rescale images 0 - 1
        fake_hr = 0.5 * fake_hr + 0.5
        imgs_lr = 0.5 * imgs_lr + 0.5

        if comp_dir:
            os.makedirs(f'images/{comp_dir}', exist_ok=True)
            # Save generated images and the high resolution originals
            titles = ['Generated', 'Low-resolution']
            fig, axs = plt.subplots(r, c)
            plt.suptitle(f'epoch: {epoch}     MSE_loss: %.5f     Perceptual_loss: %.5f' % (MSE_loss, Perceptual_loss))
            for col in range(c):
                for row, image in enumerate([fake_hr, imgs_lr]):
                    axs[row, col].imshow(image[col])
                    axs[row, col].set_title(titles[row])
                    axs[row, col].axis('off')
            fig.savefig(f"images/{comp_dir}/%d.png" % epoch)
            plt.close()

        if src_dir:
            os.makedirs(f'images/{src_dir}', exist_ok=True)
            # Save generative resolution images for comparison
            for i in range(c):
                im = Image.fromarray((255 * fake_hr[i]).astype(np.uint8))
                im.save(f"images/{src_dir}/%d_res%d.png" % (epoch, i))

        return MSE_loss, Perceptual_loss

if __name__ == '__main__':
    gan = SRGAN()
    gan.pre_train(epochs=10000, batch_size=16, sample_interval=100)
    gan.train(epochs=100000, batch_size=16, sample_interval=50)
    gan.evaluate(0, src_dir='_res', testing=True)
```

# 实验结果

## 1.  SRResNet网络预训练

**原始测试图片：**

![img](https://gitee.com/CYX12138/cloudimage/raw/master/img_2022_04/202204291804200.jpg)
![img](https://gitee.com/CYX12138/cloudimage/raw/master/img_2022_04/202204291805879.jpeg)
![img](https://gitee.com/CYX12138/cloudimage/raw/master/img_2022_04/202204291805031.jpg)

**4X bicubic 下采样测试图片：**

![img](https://gitee.com/CYX12138/cloudimage/raw/master/img_2022_04/202204291805744.jpg)
![img](https://gitee.com/CYX12138/cloudimage/raw/master/img_2022_04/202204291805550.jpg)
![img](https://gitee.com/CYX12138/cloudimage/raw/master/img_2022_04/202204291805590.jpg)

因为电脑配置有限，SRResNet的1000000轮预训练是无法接受的，经过实验，发现batch size为16时，10000轮后loss已经没有很大下降。故预训练使用10000轮，实验效果如下：

 

**Epoch=0** **：**  ![img](https://gitee.com/CYX12138/cloudimage/raw/master/img_2022_04/202204291805929.png)

 

**Epoch=1000** **：**![img](https://gitee.com/CYX12138/cloudimage/raw/master/img_2022_04/202204291805912.png)

 

**Epoch=9900** **：**

![img](https://gitee.com/CYX12138/cloudimage/raw/master/img_2022_04/202204291805521.png)

可以看出，经过SRResNet部分的训练，网络已经有了较好的超分效果，可以明显感受到分辨率的提升。

## 2.  SRGAN网络交替训练

SRGAN部分的训练按照论文所述，由生成网络和对抗网络交替训练更新：

**Epoch=0** **：**

![img](https://gitee.com/CYX12138/cloudimage/raw/master/img_2022_04/202204291806297.png)

**Epoch=20000** **：**

![img](https://gitee.com/CYX12138/cloudimage/raw/master/img_2022_04/202204291806679.png)

**Epoch=60000** **：**

![img](https://gitee.com/CYX12138/cloudimage/raw/master/img_2022_04/202204291806055.png)

**Epoch=90000** **：**

![img](https://gitee.com/CYX12138/cloudimage/raw/master/img_2022_04/202204291806978.png)

**原始测试图片：**             ![img](https://gitee.com/CYX12138/cloudimage/raw/master/img_2022_04/202204291806634.jpeg)

**SRResNet  恢复图片：**        ![img](https://gitee.com/CYX12138/cloudimage/raw/master/img_2022_04/202204291806432.jpg)

**SRGAN  恢复图片：**          ![img](https://gitee.com/CYX12138/cloudimage/raw/master/img_2022_04/202204291806632.jpg)

**原始测试图片：**             ![img](https://gitee.com/CYX12138/cloudimage/raw/master/img_2022_04/202204291806102.jpg)

**SRResNet  恢复图片：**        ![img](https://gitee.com/CYX12138/cloudimage/raw/master/img_2022_04/202204291806200.jpg)

**SRGAN  恢复图片：**          ![img](https://gitee.com/CYX12138/cloudimage/raw/master/img_2022_04/202204291806525.jpeg)

**原始测试图片：**             ![img](https://gitee.com/CYX12138/cloudimage/raw/master/img_2022_04/202204291807335.jpg)

**SRResNet  恢复图片：**        ![img](https://gitee.com/CYX12138/cloudimage/raw/master/img_2022_04/202204291807334.jpg)

**SRGAN  恢复图片：**          ![img](https://gitee.com/CYX12138/cloudimage/raw/master/img_2022_04/202204291807166.jpg)

经过比较可以发现，通过引入对抗网络，并设定基于感知损失进行参数更新，使得网络更新时更加瞄准对图像细节部分的把控，超分修复取得了非常优异的效果。

# 实验总结

本文首先介绍了图像超分相关任务，然后基于图像超分领域内的经典论文《[Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://openaccess.thecvf.com/content_cvpr_2017/papers/Ledig_Photo-Realistic_Single_Image_CVPR_2017_paper.pdf)》，对SRGAN网络的超分理论进行了相关分析，并且对网络进行了coding复现。最后在文章结尾基于本人照片进行了超分复现比较。
