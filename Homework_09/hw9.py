import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.layers import Dense, Activation, Conv2D, MaxPool2D, Conv2DTranspose, Flatten, Reshape, \
    BatchNormalization, Rescaling
import matplotlib.pyplot as plt

import os
import numpy as np
import urllib


def prepare_data(ds):
    return ds.map(to_float32).map(reshape).cache().shuffle(1000).batch(32).prefetch(32)


def to_float32(value):
    return tf.cast(value, tf.float32)


def reshape(value):
    return tf.reshape(value, (28, 28, 1))


class CandleDiscriminator(tf.keras.Model):

    def __init__(self):
        super(CandleDiscriminator, self).__init__()
        self.scaling = Rescaling(1. / 255)
        self.conv_layer1 = Conv2D(filters=64, kernel_size=3, padding='same', activation=tf.nn.relu)
        self.pooling1 = MaxPool2D(pool_size=2, strides=2)
        self.b_norm1 = BatchNormalization()
        self.activation1 = Activation(activation=tf.nn.relu)
        self.conv_layer2 = Conv2D(filters=32, kernel_size=3, padding='same', activation=tf.nn.relu)
        self.pooling2 = MaxPool2D(pool_size=2, strides=2)
        self.b_norm2 = BatchNormalization()
        self.activation2 = Activation(activation=tf.nn.relu)
        self.conv_layer3 = Conv2D(filters=16, kernel_size=3, padding='same', activation=tf.nn.relu)
        self.flatten = Flatten()
        self.out = Dense(1, activation=tf.nn.sigmoid)

    @tf.function
    def call(self, inputs):
        x = self.scaling(inputs)
        x = self.conv_layer1(x)
        x = self.pooling1(x)
        x = self.b_norm1(x)
        x = self.activation1(x)
        x = self.conv_layer2(x)
        x = self.pooling2(x)
        x = self.b_norm2(x)
        x = self.activation2(x)
        x = self.conv_layer3(x)
        x = self.flatten(x)
        return self.out(x)


class CandleGenerator(tf.keras.Model):

    def __init__(self):
        super(CandleGenerator, self).__init__()
        self.input_size = 100
        self.dense = Dense(784, activation=tf.nn.relu)  # 7 * 7 * 16
        self.reshape = Reshape((7, 7, 16))
        self.b_norm1 = BatchNormalization()
        self.activation1 = Activation(activation=tf.nn.relu)
        self.conv_t1 = Conv2DTranspose(filters=16, kernel_size=4, padding='same', strides=2, activation=tf.nn.relu)
        self.b_norm2 = BatchNormalization()
        self.activation2 = Activation(activation=tf.nn.relu)
        self.conv_t2 = Conv2DTranspose(filters=32, kernel_size=4, padding='same', strides=2, activation=tf.nn.relu)
        self.conv = Conv2D(filters=1, kernel_size=3, padding='same', activation=tf.nn.sigmoid)
        self.out = Rescaling(255)

    @tf.function
    def call(self, inputs):
        x = self.dense(inputs)
        x = self.reshape(x)
        x = self.b_norm1(x)
        x = self.activation1(x)
        x = self.conv_t1(x)
        x = self.b_norm2(x)
        x = self.activation2(x)
        x = self.conv_t2(x)
        x = self.conv(x)
        x = self.out(x)
        return x


def train(generator, discriminator, dataset, num_epochs=10, batch_size=32,
          discriminator_optimizer=tf.keras.optimizers.Adam(0.01),
          generator_optimizer=tf.keras.optimizers.Adam(0.01)):
    for epoch in range(num_epochs):
        for data in dataset:
            with tf.GradientTape() as gradient_tape, tf.GradientTape() as D_tape:
                fake_data = generator(tf.random.normal([batch_size, generator.input_size]))
                fake_data_pred = discriminator(fake_data)
                real_data_pred = discriminator(data)

                generator_loss = -tf.math.reduce_mean(tf.math.log(real_data_pred) + tf.math.log(1 - fake_data_pred))
                discriminator_loss = tf.math.reduce_mean(tf.math.log(1 - fake_data_pred))

                discriminator_gradients = D_tape.gradient(discriminator_loss, discriminator.trainable_variables)
                discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

                generator_gradients = gradient_tape.gradient(generator_loss, generator.trainable_variables)
                generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))


categories = [line.rstrip(b'\n') for line in urllib.request.urlopen(
    'https://raw.githubusercontent.com/googlecreativelab/quickdraw-dataset/master/categories.txt')]
print(categories[:10])
category = 'candle'
if not os.path.isdir('npy_files'):
    os.mkdir('npy_files')
url = f'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{category}.npy'
urllib.request.urlretrieve(url, f'npy_files/{category}.npy')

images = np.load(f'npy_files/{category}.npy')

train_ds, test_ds = np.split(images, [int(.7 * len(images))])
train_ds = tf.data.Dataset.from_tensor_slices(train_ds.values)
test_ds = tf.data.Dataset.from_tensor_slices(test_ds.values)

train_dataset = train_ds.apply(prepare_data)
test_dataset = test_ds.apply(prepare_data)

tf.keras.backend.clear_session()

generator = CandleGenerator()
discriminator = CandleDiscriminator()

train(generator, discriminator, train_dataset)

# TODO: visualize examples
for d in CandleGenerator(test_dataset.take(1)):
    plt.imshow(d[0].numpy().astype("uint8")[:, :, 0], cmap='gray')
    plt.show()
