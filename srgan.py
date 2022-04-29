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
            to_file="生成网络.png",  # 保存图片路径
            show_shapes=True,  # 是否显示形状信息
            show_layer_names=True,  # 是否显示图层名称
            rankdir="TB",  # "TB":垂直图  "LR":水平图
            expand_nested=True,  # 是否将嵌套模型展开为簇。
            dpi=96  # 图片每英寸点数。
        )

        plot_model(
            self.discriminator,  # keras模型
            to_file="判别网络.png",  # 保存图片路径
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