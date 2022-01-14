import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Conv2DTranspose, Flatten, Reshape, Rescaling
import matplotlib.pyplot as plt


def prepare_data(ds):
    return ds.map(to_float32).map(set_target).cache().shuffle(1000).prefetch(32)


def to_float32(value, target):
    return tf.cast(value, tf.float32), target


def set_target(value, target):
    return value, value


class CandleDiscriminator(tf.keras.Model):

    def __init__(self):
        super(CandleDiscriminator, self).__init__()
        self.scaling = Rescaling(1. / 255)
        self.conv_layer1 = Conv2D(filters=64, kernel_size=3, padding='same', activation=tf.nn.relu)
        self.pooling1 = MaxPool2D(pool_size=2, strides=2)
        self.conv_layer2 = Conv2D(filters=32, kernel_size=3, padding='same', activation=tf.nn.relu)
        self.pooling2 = MaxPool2D(pool_size=2, strides=2)
        self.conv_layer3 = Conv2D(filters=16, kernel_size=3, padding='same', activation=tf.nn.relu)
        self.flatten = Flatten()
        self.out = Dense(1, activation=tf.nn.sigmoid)

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


class CandleGenerator(tf.keras.Model):

    def __init__(self):
        super(CandleGenerator, self).__init__()
        self.input_size = 8
        self.dense = Dense(784, activation=tf.nn.relu)  # 7 * 7 * 16
        self.reshape = Reshape((7, 7, 16))
        self.conv_t1 = Conv2DTranspose(filters=16, kernel_size=3, padding='same', strides=2, activation=tf.nn.relu)
        self.conv_t2 = Conv2DTranspose(filters=32, kernel_size=3, padding='same', strides=2, activation=tf.nn.relu)
        self.conv = Conv2D(filters=1, kernel_size=3, padding='same', activation=tf.nn.sigmoid)
        self.out = Rescaling(255)

    @tf.function
    def call(self, inputs):
        x = self.dense(inputs)
        x = self.reshape(x)
        x = self.conv_t1(x)
        x = self.conv_t2(x)
        x = self.conv(x)
        x = self.out(x)
        return x


def train(generator, discriminator, dataset, num_epochs=10, batch_size=32,
          discriminator_optimizer=tf.keras.optimizers.Adam(0.01),
          generator_optimizer=tf.keras.optimizers.Adam(0.01)):
    for epoch in range(num_epochs):
        for data in dataset:
            with tf.GradientTape() as G_tape, tf.GradientTape() as D_tape:
                fake_data = generator(tf.random.normal([batch_size, generator.input_size]))
                fake_data_pred = discriminator(fake_data)
                real_data_pred = discriminator(data)

                generator_loss = -tf.math.reduce_mean(tf.math.log(real_data_pred) + tf.math.log(1 - fake_data_pred))
                discriminator_loss = tf.math.reduce_mean(tf.math.log(1 - fake_data_pred))

                discriminator_gradients = D_tape.gradient(discriminator_loss, discriminator.trainable_variables)
                discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

                generator_gradients = G_tape.gradient(generator_loss, generator.trainable_variables)
                generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))


tf.keras.backend.clear_session()

train_ds, test_ds = None  # TODO create dataset
train_dataset = train_ds.apply(prepare_data)
test_dataset = test_ds.apply(prepare_data)

generator = CandleGenerator()
discriminator = CandleDiscriminator()

train(generator, discriminator, train_dataset)

# TODO: visualize examples
