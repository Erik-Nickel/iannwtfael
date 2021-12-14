import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, GlobalAvgPool2D, Conv2DTranspose, Flatten, Reshape
import matplotlib.pyplot as plt
import numpy as np


def prepare_mnist_data(ds):
    return ds.map(to_float32).map(normalize).map(set_target).map(add_noise).cache().shuffle(1000).batch(16).prefetch(20)


def to_float32(value, target):
    return tf.cast(value, tf.float32), target


# from range [0, 255] to [-1, 1]
def normalize(value, target):
    return (value / 128.) - 1., target


def set_target(value, target):
    return value, value


def add_noise(value, target):
    return value, target  # TODO: add noise to value


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
        self.conv_layer1 = Conv2D(filters=64, kernel_size=3, padding='same', activation=tf.nn.relu)
        self.flatten = Flatten()  # or GlobalAvgPool2D()
        self.out = Dense(128, activation=tf.nn.relu)

    @tf.function
    def call(self, inputs):
        return self.out(self.flatten(self.conv_layer1(inputs)))


class DeNoiseDecoderModel(tf.keras.Model):

    def __init__(self):
        super(DeNoiseDecoderModel, self).__init__()
        self.dense = Dense(50176, activation=tf.nn.relu)  # 28 * 28 * 64 = 50176
        self.reshape = Reshape((28, 28, 64))
        self.conv_t = Conv2DTranspose(filters=64, kernel_size=3, padding='same', activation=tf.nn.relu)
        self.out = Conv2D(filters=1, kernel_size=3, padding='same', activation=tf.nn.sigmoid)

    @tf.function
    def call(self, inputs):
        x = self.dense(inputs)
        x = self.reshape(x)
        x = self.conv_t(x)
        x = self.out(x)
        return x


@tf.function
def train_step(model, input, target, loss_function, optimizer):
    with tf.GradientTape() as tape:
        prediction = model(input)
        loss = loss_function(target, prediction)
        gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


def test(model, test_data, loss_function):
    test_accuracy_aggregator = []
    test_loss_aggregator = []
    for (input, target) in test_data:
        prediction = model(input)
        sample_test_loss = loss_function(target, prediction)
        sample_test_accuracy = np.argmax(target, axis=1) == np.argmax(prediction, axis=1)
        sample_test_accuracy = np.mean(sample_test_accuracy)
        test_loss_aggregator.append(sample_test_loss.numpy())
        test_accuracy_aggregator.append(np.mean(sample_test_accuracy))
    test_loss = tf.reduce_mean(test_loss_aggregator)
    test_accuracy = tf.reduce_mean(test_accuracy_aggregator)
    return test_loss, test_accuracy


tf.keras.backend.clear_session()

train_ds, test_ds = tfds.load('MNIST', split=['train', 'test'], as_supervised=True)
train_dataset = train_ds.apply(prepare_mnist_data)
test_dataset = test_ds.apply(prepare_mnist_data)

num_epochs = 10
learning_rate = 0.01  # TODO: Parameter tuning

model = DeNoiseAutoEncoderModel()
cross_entropy_loss = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate)

train_losses = []
test_losses = []
test_accuracies = []

test_loss, test_accuracy = test(model, test_dataset, cross_entropy_loss)
test_losses.append(test_loss)
test_accuracies.append(test_accuracy)

train_loss, _ = test(model, train_dataset, cross_entropy_loss)
train_losses.append(train_loss)

for epoch in range(num_epochs):
    print(f'Epoch: {str(epoch)} starting with accuracy {test_accuracies[-1]}')
    epoch_loss_agg = []
    for input, target in train_dataset:
        train_loss = train_step(model, input, target, cross_entropy_loss, optimizer)
        epoch_loss_agg.append(train_loss)
    train_losses.append(tf.reduce_mean(epoch_loss_agg))
    test_loss, test_accuracy = test(model, test_dataset, cross_entropy_loss)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)

plt.figure()
plt.plot(train_losses, label="training")
plt.plot(test_losses, label="test")
plt.plot(test_accuracies, label="test accuracy")
plt.xlabel("Training steps")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
