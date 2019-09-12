# -*- coding: utf-8 -*-
# !@time: 19-5-20 下午9:10
# !@author: superMC @email: 18758266469@163.com
# !@fileName: C_gan.py

# -*- coding: utf-8 -*-
# !@time: 19-5-19 下午11:34
# !@author: superMC @email: 18758266469@163.com
# !@fileName: 2.0-self.py.py

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.layers import BatchNormalization, Embedding, ZeroPadding2D
from tensorflow.python.keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
from tensorflow.python.keras.layers.convolutional import UpSampling2D, Conv2D
from tensorflow.python.keras.models import Sequential, Model
from tqdm.autonotebook import tqdm
from tensorflow.python.keras.utils import plot_model

auto_growth = tf.config.gpu.set_per_process_memory_growth(True)

TRAIN_BUFFER = 60000
TEST_BUFFER = 10000
BATCH_SIZE = 512
DIMS = (28, 28, 1)
N_TRAIN_BATCHES = int(TRAIN_BUFFER / BATCH_SIZE)
N_TEST_BATCHES = int(TEST_BUFFER / BATCH_SIZE)


def get_dataset():
    (train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()

    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype(np.float32) / 127.5 - 1.
    train_labels = train_labels.reshape(-1, 1)
    return train_images, train_labels


class Gan():
    def __init__(self, units=100):
        self.shape = (28, 28, 1)
        self.units = units
        self.num_classes = 10
        self.mode_file = 'models/acGan_gen.ckpt'
        self.img_file = 'imgs/acGan'
        optimizer = tf.keras.optimizers.Adam(0.0002, beta_1=0.5)
        losses = ['binary_crossentropy', 'sparse_categorical_crossentropy']
        self.disc = self.build_disc_model()
        self.disc.compile(optimizer=optimizer, loss=losses,
                          metrics=['accuracy'])

        # build combined model
        self.gen = self.build_gen_model()
        noise = Input(shape=(self.units,))
        label = Input(shape=(1,))
        img_gen = self.gen([noise, label])

        # 建立数据传输的桥梁
        self.disc.trainable = False
        valid, logit = self.disc(img_gen)

        self.combined = Model([noise, label], [valid, logit])
        self.combined.compile(optimizer=optimizer, loss=losses)

    def build_gen_model(self):
        model = Sequential()
        model.add(Dense(7 * 7 * 128, activation='relu', input_dim=self.units))
        model.add(Reshape(target_shape=(7, 7, 128)))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(128, 3, 1, padding='same', activation='relu'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(64, 3, 1, padding='same', activation='relu'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(self.shape[2], 3, padding='same', activation='tanh'))
        model.summary()

        noise = Input(shape=(self.units,))
        label = Input(shape=(1,), dtype='int32')
        label_embedding = Flatten()(Embedding(self.num_classes, 100)(label))
        model_input = multiply([noise, label_embedding])
        img_gen = model(model_input)

        return Model([noise, label], img_gen)

    def build_disc_model(self):
        model = Sequential()

        model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=self.shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.summary()

        img = Input(shape=self.shape)

        # Extract feature representation
        features = model(img)

        # Determine validity and label of the image
        validity = Dense(1, activation="sigmoid")(features)
        label = Dense(self.num_classes, activation="softmax")(features)

        return Model(img, [validity, label])

    def plot_reconstruction(self, model, noise_numpy, noise_label, n_cols=5, n_rows=2, dim=2, name='model'):

        example_data_reconstructed = model.predict([noise_numpy, noise_label])
        fig, axs = plt.subplots(ncols=n_cols, nrows=n_rows, figsize=(n_cols * dim, n_rows * dim))
        for col in range(n_cols):
            for row in range(n_rows):
                # 第一个维度为batch_size
                axs[row, col].matshow(example_data_reconstructed[col * n_rows + row].squeeze(), cmap=plt.cm.Greys,
                                      vmin=-1,
                                      vmax=1)  # 因为使用的是tanh所以改为-1. ~ 1
                axs[row, col].axis('off')  # 此处不会生成坐标轴
        if not os.path.exists(self.img_file):
            os.makedirs(self.img_file)
        plt.savefig(self.img_file + '/' + name + '.png')
        plt.close()

    def train(self, itera, batch_size, train_dataset, train_labels, show_itera=500):
        noise = np.random.normal(0, 1, (batch_size, self.units))
        noise_label = np.random.randint(0, 10, (batch_size, 1))
        valid = np.ones(shape=(batch_size, 1))
        fake = np.zeros(shape=(batch_size, 1))

        for i in tqdm(range(itera)):
            # random choose data
            idx = np.random.randint(0, train_dataset.shape[0], batch_size)
            imgs = train_dataset[idx]
            labels = train_labels[idx]
            # random generate noise
            z_samp = np.random.normal(0, 1, (batch_size, self.units))
            l_samp = np.random.randint(0, 10, (batch_size, 1))

            img_gen = self.gen.predict([z_samp, l_samp])  # 只预测
            # img_gen = self.gen(z_samp)  # 会训练!

            disc_real_loss = self.disc.train_on_batch(imgs, [valid, labels])
            disc_fake_loss = self.disc.train_on_batch(img_gen, [fake, l_samp])
            disc_loss = 0.5 * np.add(disc_real_loss, disc_fake_loss)

            gen_loss = self.combined.train_on_batch([z_samp, l_samp], [valid, l_samp])
            if i % show_itera == 0:
                print("%d [D loss: %f, acc.: %.2f%%, op_acc: %.2f%%] [G loss: %f]" % (
                    i, disc_loss[0], 100 * disc_loss[3], 100 * disc_loss[4], gen_loss[0]))
                # total_loss 2分类(真假)损失 10分类(0-9)损失 判断真假正确率 判断0-9 正确率 loss_weights 1:1
                name = 'acGan%d' % i
                self.plot_reconstruction(self.gen, noise, noise_label, name=name)
        self.gen.save(self.mode_file)
        plot_model(self.combined, self.mode_file + '.png')

    def test(self, n_cols=5, n_rows=4, dim=2, label=0):
        model = keras.models.load_model(self.mode_file)
        noise = np.random.normal(0, 1, (n_cols * n_rows, self.units))

        l_laels = np.ones(shape=(n_cols * n_rows, 1), dtype='int32') * label

        example_data_reconstructed = model.predict([noise, l_laels])

        # fig, axs = plt.subplots(ncols=n_cols, nrows=n_rows, figsize=(n_cols * dim, n_rows * dim))
        # for col in range(n_cols):
        #     for row in range(n_rows):
        #         # 第一个维度为batch_size
        #         axs[row, col].matshow(example_data_reconstructed[col * n_rows + row].squeeze(), cmap=plt.cm.Greys,
        #                               vmin=-1,
        #                               vmax=1)  # 因为使用的是tanh所以改为-1. ~ 1
        #         axs[row, col].axis('off')  # 此处不会生成坐标轴
        '''反色'''
        fig, axs = plt.subplots(ncols=n_cols, nrows=n_rows, figsize=(n_cols * dim, n_rows * dim))
        cnt = 0
        for i in range(n_rows):
            for j in range(n_cols):
                axs[i, j].imshow(example_data_reconstructed[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1

        plt.show()
        plt.close()
        plot_model(model, self.mode_file + '.png')


if __name__ == '__main__':
    train_dataset, train_labels = get_dataset()
    gan = Gan(units=100)
    gan.train(20000, 512, train_dataset, train_labels)
    # gan.test(label=0)
