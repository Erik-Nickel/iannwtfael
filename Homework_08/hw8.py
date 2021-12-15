import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Conv2DTranspose, Flatten, Reshape, Rescaling
import matplotlib.pyplot as plt
import numpy as np


def prepare_mnist_data(ds):
    return ds.map(to_float32).map(set_target).map(add_noise).cache().shuffle(1000).batch(32).prefetch(32)


def to_float32(value, target):
    return tf.cast(value, tf.float32), target


def set_target(value, target):
    return value, value


def add_noise(value, target):
    return value + 2 * np.random.normal(loc=0., scale=1., size=value.shape), target


class DeNoiseAutoEncoderModel(tf.keras.Model):

    def __init__(self):
        super(DeNoiseAutoEncoderModel, self).__init__()
        self.encoder = DeNoiseEncoderModel()
        self.decoder = DeNoiseDecoderModel()

    @tf.function
    def call(self, inputs):
        return self.decoder(self.encoder(inputs))


class DeNoiseEncoderModel(tf.keras.Model):

    def __init__(self):
        super(DeNoiseEncoderModel, self).__init__()
        self.scaling = Rescaling(1. / 255)
        self.conv_layer1 = Conv2D(filters=32, kernel_size=3, padding='same', activation=tf.nn.relu)
        self.pooling1 = MaxPool2D(pool_size=2, strides=2)
        self.conv_layer2 = Conv2D(filters=32, kernel_size=3, padding='same', activation=tf.nn.relu)
        self.pooling2 = MaxPool2D(pool_size=2, strides=2)
        self.conv_layer3 = Conv2D(filters=16, kernel_size=3, padding='same', activation=tf.nn.relu)
        self.flatten = Flatten()
        self.out = Dense(16, activation=tf.nn.relu)

    @tf.function
    def call(self, inputs):
        x = self.scaling(inputs)
        x = self.conv_layer1(x)
        x = self.pooling1(x)
        x = self.conv_layer2(x)
        x = self.pooling2(x)
        x = self.conv_layer3(x)
        x = self.flatten(x)
        return self.out(x)


class DeNoiseDecoderModel(tf.keras.Model):

    def __init__(self):
        super(DeNoiseDecoderModel, self).__init__()
        self.dense = Dense(784, activation=tf.nn.relu)  # 7 * 7 * 16 = 784
        self.reshape = Reshape((7, 7, 16))
        self.conv_t1 = Conv2DTranspose(filters=16, kernel_size=2, strides=2, activation=tf.nn.relu)
        self.conv_layer1 = Conv2D(filters=32, kernel_size=3, padding='same', activation=tf.nn.relu)
        self.conv_t2 = Conv2DTranspose(filters=32, kernel_size=2, strides=2, activation=tf.nn.relu)
        self.conv = Conv2D(filters=1, kernel_size=3, padding='same', activation=tf.nn.sigmoid)
        self.out = Rescaling(255)

    @tf.function
    def call(self, inputs):
        x = self.dense(inputs)
        x = self.reshape(x)
        x = self.conv_t1(x)
        x = self.conv_layer1(x)
        x = self.conv_t2(x)
        x = self.conv(x)
        x = self.out(x)
        return x


tf.keras.backend.clear_session()

train_ds, test_ds = tfds.load('MNIST', split=['train', 'test'], as_supervised=True)
train_dataset = train_ds.apply(prepare_mnist_data)
test_dataset = test_ds.apply(prepare_mnist_data)

num_epochs = 5
learning_rate = 0.001

model = DeNoiseAutoEncoderModel()
loss = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate)
model.compile(optimizer=optimizer, loss=loss)

images = None
for images, labels in test_dataset.take(1):
    images = images

for i in range(num_epochs):
    plt.imshow(images[0].numpy().astype("uint8")[:, :, 0], cmap='gray')
    plt.show()
    x = model(images)
    plt.imshow(x[0].numpy().astype("uint8")[:, :, 0], cmap='gray')
    plt.show()
    model.fit(train_dataset, epochs=1)

plt.imshow(images[0].numpy().astype("uint8")[:, :, 0], cmap='gray')
plt.show()
x = model(images)
plt.imshow(x[0].numpy().astype("uint8")[:, :, 0], cmap='gray')
plt.show()
