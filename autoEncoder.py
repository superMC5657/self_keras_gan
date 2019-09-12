from tensorflow.python import keras
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.keras.utils import plot_model

tf.config.gpu.set_per_process_memory_growth(True)


def get_dataset():
    (train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype(np.float32) / 255
    test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype(np.float32) / 255.0
    return train_images, test_images


def model(units):
    input_img = keras.layers.Input(shape=(28, 28, 1))
    x = keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation="relu")(
        input_img)  # (28-3)/2+1 = 14 向下取整
    x = keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation="relu")(x)
    x = keras.layers.Flatten()(x)
    encode = keras.layers.Dense(units=units)(x)

    x = keras.layers.Dense(units=7 * 7 * 64, activation="relu")(encode)
    x = keras.layers.Reshape(target_shape=(7, 7, 64))(x)
    x = keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=(2, 2), padding="SAME", activation="relu")(x)
    x = keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=(2, 2), padding="SAME", activation="relu")(x)
    decode = keras.layers.Conv2DTranspose(filters=1, kernel_size=3, strides=(1, 1), padding="SAME",
                                          activation="sigmoid")(x)

    autoencoder = keras.Model(input_img, decode)
    autoencoder.compile(optimizer='adam', loss='mse')

    return autoencoder


def plot_reconstruction(model, example_data, nex=5, dim=3):
    example_data_reconstructed = model.predict(example_data)

    nums = int(example_data.shape[0] / nex)
    for i in range(nums):
        fig, axs = plt.subplots(ncols=nex, nrows=2, figsize=(dim * nex, dim * 2))
        for exi in range(nex):
            # 第一个维度为batch_size
            index = 5 * i + exi
            axs[0, exi].matshow(example_data[index].squeeze(), cmap=plt.cm.Greys, vmin=0, vmax=1)  # 灰度图片 摇降成2维
            axs[1, exi].matshow(example_data_reconstructed[index].squeeze(), cmap=plt.cm.Greys, vmin=0, vmax=1)

        for ax in axs.flatten():
            ax.axis('off')  # 此处不会有坐标
        plt.savefig('imgs/autoEncoder/ae_%d.jpg' % (i))


def Train():
    autoencoder = model(64)
    train_dataset, test_dataset = get_dataset()
    autoencoder.fit(train_dataset, train_dataset, batch_size=256, epochs=5, shuffle=True, )
    autoencoder.save('models/autoEncoder.ckpt')
    plot_reconstruction(autoencoder, test_dataset[:5])


def Test():
    train_dataset, test_dataset = get_dataset()
    autoencoder = keras.models.load_model('models/autoEncoder.ckpt')  # 保存了结构和参数
    plot_reconstruction(autoencoder, test_dataset[:10])
    plot_model(autoencoder, 'models/autoEncoder.png')


if __name__ == '__main__':
    Test()
