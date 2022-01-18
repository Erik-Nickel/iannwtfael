import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation, Conv2D, Conv2DTranspose, Flatten, Reshape, BatchNormalization, \
    Rescaling, Dropout
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
        self.scaling = Rescaling(1. / 127.5, offset=-1)
        self.conv_layer1 = Conv2D(filters=32, kernel_size=3, padding='same', activation=tf.nn.relu)
        self.conv_layer1 = Conv2D(filters=32, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)
        self.dropout1 = Dropout(0.2)
        self.conv_layer2 = Conv2D(filters=64, kernel_size=3, padding='same', activation=tf.nn.relu)
        self.conv_layer2 = Conv2D(filters=64, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)
        self.dropout2 = Dropout(0.2)
        self.conv_layer3 = Conv2D(filters=128, kernel_size=3, padding='same', activation=tf.nn.relu)
        self.conv_layer3 = Conv2D(filters=128, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)
        self.dropout3 = Dropout(0.2)
        self.flatten = Flatten()
        self.out = Dense(1, activation=tf.nn.sigmoid)  # activation=None)

    @tf.function
    def call(self, inputs, is_training=False):
        x = self.scaling(inputs)
        x = self.conv_layer1(x)
        x = self.dropout1(x, training=is_training)
        x = self.conv_layer2(x)
        x = self.dropout2(x, training=is_training)
        x = self.conv_layer3(x)
        x = self.dropout3(x, training=is_training)
        x = self.flatten(x)
        return self.out(x)


class CandleGenerator(tf.keras.Model):

    def __init__(self):
        super(CandleGenerator, self).__init__()
        self.input_size = 100
        self.dense = Dense(7 * 7 * 256)
        self.reshape = Reshape((7, 7, 256))
        self.b_norm1 = BatchNormalization()
        self.activation1 = Activation(activation=tf.nn.relu)
        self.conv1 = Conv2D(filters=256, kernel_size=3, padding='same')
        self.b_norm2 = BatchNormalization()
        self.activation2 = Activation(activation=tf.nn.relu)
        self.conv_t1 = Conv2DTranspose(filters=128, kernel_size=4, padding='same', strides=2)
        self.conv2 = Conv2D(filters=128, kernel_size=3, padding='same')
        self.b_norm3 = BatchNormalization()
        self.activation3 = Activation(activation=tf.nn.relu)
        self.conv_t2 = Conv2DTranspose(filters=64, kernel_size=4, padding='same', strides=2)
        self.conv3 = Conv2D(filters=64, kernel_size=3, padding='same')
        self.b_norm4 = BatchNormalization()
        self.activation4 = Activation(activation=tf.nn.relu)
        self.conv = Conv2D(filters=1, kernel_size=3, padding='same', activation=tf.nn.sigmoid)  # activation=tf.nn.tanh
        self.out = Rescaling(255)

    @tf.function
    def call(self, inputs):
        x = self.dense(inputs)
        x = self.reshape(x)
        x = self.b_norm1(x)
        x = self.activation1(x)
        x = self.conv1(x)
        x = self.b_norm2(x)
        x = self.activation2(x)
        x = self.conv_t1(x)
        x = self.conv2(x)
        x = self.b_norm3(x)
        x = self.activation3(x)
        x = self.conv_t2(x)
        x = self.conv3(x)
        x = self.b_norm4(x)
        x = self.activation4(x)
        x = self.conv(x)
        x = self.out(x)
        return x


def train(generator, discriminator, dataset, num_epochs=50,
          discriminator_optimizer=tf.keras.optimizers.Adam(0.0001),
          generator_optimizer=tf.keras.optimizers.Adam(0.0001)):
    b = generator(tf.random.normal([1, generator.input_size]))
    plt.imshow(b[0].numpy().astype("uint8")[:, :, 0], cmap='gray')
    plt.show()
    for epoch in range(num_epochs):
        print('epoch: ', epoch)
        disc = []
        gen = []
        for data in dataset:
            with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:
                batch_size = data.get_shape()[0]
                fake_data = generator(tf.random.normal([batch_size, generator.input_size]))
                fake_data_pred = discriminator(fake_data, True)
                real_data_pred = discriminator(data, True)

                loss_fun = tf.keras.losses.BinaryCrossentropy()

                generator_loss = loss_fun(tf.ones_like(fake_data_pred), fake_data_pred)
                discriminator_loss = (loss_fun(tf.ones_like(real_data_pred), real_data_pred) + loss_fun(
                    tf.zeros_like(fake_data_pred), fake_data_pred))

                disc.append(discriminator_loss)
                gen.append(generator_loss)

                discriminator_gradients = discriminator_tape.gradient(discriminator_loss,
                                                                      discriminator.trainable_variables)
                discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

                generator_gradients = generator_tape.gradient(generator_loss, generator.trainable_variables)
                generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
        print('disc: ', np.mean(disc))
        print('gen: ', np.mean(gen))
        disc = []
        gen = []
        b = generator(tf.random.normal([1, generator.input_size]))
        plt.imshow(b[0].numpy().astype("uint8")[:, :, 0], cmap='gray')
        plt.show()


category = 'candle'
if not os.path.isdir('npy_files'):
    os.mkdir('npy_files')
url = f'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{category}.npy'
urllib.request.urlretrieve(url, f'npy_files/{category}.npy')
images = np.load(f'npy_files/{category}.npy')

train_ds = tf.data.Dataset.from_tensor_slices(images)
train_dataset = train_ds.apply(prepare_data)

tf.keras.backend.clear_session()

generator = CandleGenerator()
discriminator = CandleDiscriminator()
train(generator, discriminator, train_dataset)
