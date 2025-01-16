import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential  # 确保Sequential导入
from tensorflow.keras.layers import Dense, Reshape, Input, Flatten, Dropout, BatchNormalization, Activation, UpSampling2D, Conv2D, LeakyReLU, ZeroPadding2D
import matplotlib.pyplot as plt
from functools import partial
from tensorflow.keras.layers import Lambda
from tensorflow.keras import layers

class WGANGP:
    def __init__(self, img_shape=(64, 64, 3), latent_dim=100):
        self.img_shape = img_shape
        self.latent_dim = latent_dim

        # Optimizer settings from the original WGAN-GP paper
        self.n_critic = 5
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.00005)

        # Build the generator and critic models
        self.generator = self.build_generator()
        self.critic = self.build_critic()

        # Create the critic model with gradient penalty
        real_img = tf.keras.Input(shape=self.img_shape)
        z_disc = tf.keras.Input(shape=(self.latent_dim,))
        fake_img = self.generator(z_disc)
        fake = self.critic(fake_img)
        valid = self.critic(real_img)

        # Create interpolated image as symbolic tensor
        interpolated_img = self.random_weighted_average([real_img, fake_img])

        # Initialize the critic model
        self.critic_model = tf.keras.Model(inputs=[real_img, z_disc],
                                           outputs=[valid, fake])
        self.critic_model.compile(loss=[self.wasserstein_loss, self.wasserstein_loss],
                                   optimizer=optimizer)

        # Create the generator model
        self.critic.trainable = False
        self.generator.trainable = True

        z_gen = tf.keras.Input(shape=(self.latent_dim,))
        img = self.generator(z_gen)
        valid = self.critic(img)
        self.generator_model = tf.keras.Model(z_gen, valid)
        self.generator_model.compile(loss=self.wasserstein_loss, optimizer=optimizer)

    def wasserstein_loss(self, y_true, y_pred):
        return tf.reduce_mean(y_true * y_pred)

    @tf.function
    def get_gradient_penalty_loss(self, real_img, fake_img):
        """
        Gradient penalty loss function.
        """
        interpolated_img = self.random_weighted_average([real_img, fake_img])
        with tf.GradientTape() as tape:
            tape.watch(interpolated_img)  # Watch the tensor directly
            validity_interpolated = self.critic(interpolated_img)
        gradients = tape.gradient(validity_interpolated, interpolated_img)
        grad_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
        gradient_penalty = tf.reduce_mean((grad_norm - 1.0) ** 2)
        return gradient_penalty

    from tensorflow.keras import layers
    import tensorflow as tf

    def random_weighted_average(self, images):
        real_img, fake_img = images
        # 打印真实输入图像的形状
        print("Real Image Shape: ", real_img.shape)  # 输出真实输入图像的形状

        # 使用 Lambda 层获取批次大小
        batch_size = layers.Lambda(lambda x: tf.shape(x)[0])(real_img)

        # 生成 alpha
        alpha = tf.random.uniform(shape=[batch_size, 1, 1, 1], minval=0.0, maxval=1.0)

        # 计算加权平均
        interpolated_img = alpha * real_img + (1 - alpha) * fake_img

        return interpolated_img


    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def build_generator(self):
        model = Sequential()
        noise = Input(shape=(self.latent_dim,))
        model.add(Dense(128 * (self.img_shape[0] // 4) * (self.img_shape[1] // 4), activation="relu"))
        model.add(Reshape((self.img_shape[0] // 4, self.img_shape[1] // 4, 128)))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(self.img_shape[2], kernel_size=4, padding="same"))
        model.add(Activation("tanh"))
        noise = Input(shape=(self.latent_dim,))
        img = model(noise)
        return Model(noise, img)

    def build_critic(self):
        model = Sequential()
        model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(negative_slope=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(negative_slope=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(negative_slope=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(negative_slope=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1))
        img = Input(shape=self.img_shape)
        validity = model(img)
        return Model(img, validity)

    def load_data(self, data_path, target_shape=None):
        images = []
        for file_name in os.listdir(data_path):
            img_path = os.path.join(data_path, file_name)
            img = load_img(img_path, target_size=target_shape, color_mode="rgb" if self.img_shape[2] == 3 else "grayscale")
            img = img_to_array(img)
            images.append(img)
        images = np.array(images)
        images = (images.astype(np.float32) - 127.5) / 127.5
        return images

    def train_step(self, real_images, z_disc):
            """
            A single training step for WGAN-GP.
            """
            fake_images = self.generator(z_disc)
            real_validity = self.critic(real_images)
            fake_validity = self.critic(fake_images)

            # Calculate gradient penalty
            gradient_penalty = self.get_gradient_penalty_loss(real_images, fake_images)

            # Critic loss
            critic_loss = tf.reduce_mean(fake_validity - real_validity) + gradient_penalty

            # Backprop for critic
            critic_grads = tf.gradients(critic_loss, self.critic.trainable_variables)
            self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

            return critic_loss

    def sample_images(self, epoch, save_path):
        os.makedirs(save_path, exist_ok=True)
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c, figsize=(10, 10))
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, :] if self.img_shape[2] == 3 else gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig(f"{save_path}/image_{epoch}.png")
        plt.close()

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
# from wgan_gp import WGANGP  # 假设WGANGP类保存在wgan_gp.py文件中

# 训练参数配置
data_path = "小波时频/1/"  # 替换为你的数据集路径
save_path = "小波时频/11/"  # 生成图片保存路径
epochs = 10000  # 训练的总周期
batch_size = 64  # 每批次的样本数
sample_interval = 100  # 每隔多少轮保存一次生成的图像

# 载入数据
def load_data(data_path, img_shape=(64, 64, 3)):
    """载入图片数据并进行预处理"""
    images = []
    for filename in os.listdir(data_path):
        img_path = os.path.join(data_path, filename)
        try:
            img = load_img(img_path, target_size=img_shape)  # 读取并调整图片尺寸
            img = img_to_array(img) / 127.5 - 1.0  # 归一化处理
            images.append(img)
        except Exception as e:
            print(f"Error loading image {filename}: {e}")
    return np.array(images)

# 初始化WGAN-GP模型
wgan_gp = WGANGP(img_shape=(64, 64, 3), latent_dim=100)  # 设置图片大小和潜在维度

# 训练过程
for epoch in range(epochs):
    # 加载一个批次的真实图像
    real_images = load_data(data_path, img_shape=(64, 64, 3))

    # 随机选择噪声向量作为潜在空间的输入
    z_disc = np.random.normal(0, 1, (batch_size, 100))

    # 训练判别器 (Critic)
    critic_loss = wgan_gp.train_step(real_images, z_disc)

    # 每训练一定周期后保存生成的图像
    if epoch % sample_interval == 0:
        print(f"Epoch: {epoch}, Critic Loss: {critic_loss}")
        # 保存生成的图片
        wgan_gp.sample_images(epoch, save_path)

    # 训练生成器 (Generator) 每隔 n_critic 次
    if epoch % wgan_gp.n_critic == 0:
        z_gen = np.random.normal(0, 1, (batch_size, 100))  # 随机噪声生成器输入
        g_loss = wgan_gp.generator_model.train_on_batch(z_gen, np.ones((batch_size, 1)))  # 训练生成器
        print(f"Epoch: {epoch}, Generator Loss: {g_loss}")

    # 每隔一定周期保存模型（如果需要的话）
    if epoch % 500 == 0:
        wgan_gp.generator.save(os.path.join(save_path, f'generator_epoch_{epoch}.h5'))
        wgan_gp.critic.save(os.path.join(save_path, f'critic_epoch_{epoch}.h5'))

