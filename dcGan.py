# -*- coding: utf-8 -*-
# !@time: 19-5-19 下午11:34
# !@author: superMC @email: 18758266469@163.com
# !@fileName: 2.0-self.py.py


import tensorflow as tf
from tqdm.autonotebook import tqdm
from tensorflow.python import keras
from tensorflow.python.keras import Sequential, Model
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.layers import Dense, Conv2D, Conv2DTranspose, Flatten, Input, Reshape
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

    return train_images


class Gan():
    def __init__(self, units):

        self.units = units
        self.gen = self.build_gen_model()
        self.disc = self.build_disc_model()
        optimizer = tf.keras.optimizers.Adam(0.0002, beta_1=0.5)

        self.disc.compile(optimizer=optimizer, loss='binary_crossentropy',
                          metrics=['accuracy'])

        # build combined model
        z = Input(shape=(self.units,))
        img_gen = self.gen(z)

        self.disc.trainable = False
        validity = self.disc(img_gen)

        # No Warning
        '''        base_disc = self.build_disc_model()
        self.disc = Model(inputs=base_disc.inputs, outputs=base_disc.outputs)
        optimizer = tf.keras.optimizers.Adam(0.0002, beta_1=0.5)

        self.disc.compile(optimizer=optimizer, loss='binary_crossentropy',
                          metrics=['accuracy'])

        # build combined model
        z = Input(shape=(self.units,))
        img_gen = self.gen(z)

        disc_freeze = Model(inputs=base_disc.inputs, outputs=base_disc.outputs)
        disc_freeze.trainable = False
        validity = disc_freeze(img_gen)'''
        self.combined = Model(z, validity)
        # self.combined = Sequential(self.gen, self.disc)
        self.combined.compile(optimizer=optimizer, loss='binary_crossentropy')

    def build_gen_model(self):
        model = Sequential()
        model.add(Dense(7 * 7 * 128, activation='relu', input_dim=self.units))
        model.add(Reshape(target_shape=(7, 7, 128)))
        model.add(Conv2DTranspose(64, 3, 2, 'same', activation='relu'))
        model.add(Conv2DTranspose(32, 3, 2, 'same', activation='relu'))
        model.add(Conv2DTranspose(1, 3, 1, 'same', activation='tanh'))
        model.summary()

        # z = Input(shape=(self.units,))
        # img_gen = model(z)
        # return Model(z, img_gen)
        return model

    def build_disc_model(self):
        model = Sequential()
        model.add(Conv2D(32, 3, 2, 'same', activation='relu', input_shape=(28, 28, 1)))
        model.add(Conv2D(64, 3, 2, 'same', activation='relu'))
        model.add(Flatten())
        model.add(Dense(units=1, activation='sigmoid'))
        model.summary()

        # img = Input(shape=(28, 28, 1))
        # logit = model(img)
        # return Model(img, logit)
        return model

    def plot_reconstruction(self, model, noise_numpy, nex=5, dim=2, name='model'):
        example_data_reconstructed = model.predict(noise_numpy)
        fig, axs = plt.subplots(ncols=nex, nrows=1, figsize=(dim * nex, dim * 2))
        for exi in range(nex):
            # 第一个维度为batch_size
            axs[exi].matshow(example_data_reconstructed[exi].squeeze(), cmap=plt.cm.Greys, vmin=-1,
                             vmax=1)  # 因为使用的是tanh所以改为-0.5 ~ 0.5
            axs[exi].axis('off')  # 此处不会生成坐标轴
        plt.savefig('imgs/dcGan/' + name + '.jpg')
        plt.close()

    def train(self, itera, batch_size, train_dataset, show_itera=500):
        noise = np.random.normal(0, 1, (batch_size, self.units))
        valid = np.ones(shape=(batch_size, 1))
        fake = np.zeros(shape=(batch_size, 1))

        for i in tqdm(range(itera)):
            idx = np.random.randint(0, train_dataset.shape[0], batch_size)
            imgs = train_dataset[idx]
            z_samp = np.random.normal(0, 1, (batch_size, self.units))
            img_gen = self.gen.predict(z_samp)  # 只预测
            # img_gen = self.gen(z_samp)  # 会训练!

            disc_real_loss = self.disc.train_on_batch(imgs, valid)
            disc_fake_loss = self.disc.train_on_batch(img_gen, fake)
            disc_loss = 0.5 * np.add(disc_real_loss, disc_fake_loss)

            gen_loss = self.combined.train_on_batch(z_samp, valid)
            if i % show_itera == 0:
                print("\n%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (i, disc_loss[0], 100 * disc_loss[1], gen_loss))
                name = 'dcGan%d' % i

                self.plot_reconstruction(self.gen, noise, name=name)
        self.gen.save('models/dcGan_gen.ckpt')
        plot_model(self.combined, 'models/dcGan_gen.png')

    def test(self, n_cols=5, n_rows=4):
        model = keras.models.load_model('models/dcGan_gen.ckpt')
        noise = np.random.normal(0, 1, (n_cols * n_rows, self.units))
        example_data_reconstructed = model.predict(noise)
        fig, axs = plt.subplots(ncols=n_cols, nrows=n_rows, figsize=(n_cols, n_rows))
        for col in range(n_cols):
            for row in range(n_rows):
                # 第一个维度为batch_size
                axs[row, col].matshow(example_data_reconstructed[col * n_rows + row].squeeze(), cmap=plt.cm.Greys,
                                      vmin=-1,
                                      vmax=1)  # 因为使用的是tanh所以改为-0.5 ~ 0.5
                axs[row, col].axis('off')  # 此处不会生成坐标轴
        plt.show()
        plt.close()
        plot_model(model, 'models/dcGan_gen.png')


if __name__ == '__main__':
    train_dataset = get_dataset()
    gan = Gan(units=100)
    gan.train(20000, 128, train_dataset)
    # gan.test()
