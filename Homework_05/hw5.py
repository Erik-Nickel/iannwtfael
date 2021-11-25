import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import GlobalAvgPool2D
from keras.regularizers import l2
import matplotlib.pyplot as plt
import numpy as np


def prepare_mnist_data(ds):
    return ds.map(to_float32).map(normalize).map(to_one_hot).cache().shuffle(1000).batch(32).prefetch(20)


def to_float32(value, target):
    return tf.cast(value, tf.float32), target


# from range [0, 255] to [-1, 1]
def normalize(value, target):
    return (value / 128.) - 1., target


def to_one_hot(value, target):
    return value, tf.one_hot(target, depth=10)


class FashionMnistConvModel(tf.keras.Model):

    def __init__(self):
        super(FashionMnistConvModel, self).__init__()
        reg = l2(0.01)
        self.conv_layer1 = Conv2D(filters=24, kernel_size=3, padding='same', activation=tf.nn.relu,
                                  kernel_regularizer=reg, bias_regularizer=reg, activity_regularizer=reg)
        self.conv_layer2 = Conv2D(filters=24, kernel_size=3, padding='same', activation=tf.nn.relu,
                                  kernel_regularizer=reg, bias_regularizer=reg, activity_regularizer=reg)
        self.pooling1 = MaxPool2D(pool_size=2, strides=2)
        self.conv_layer3 = Conv2D(filters=48, kernel_size=3, padding='same', activation=tf.nn.relu,
                                  kernel_regularizer=reg, bias_regularizer=reg, activity_regularizer=reg)
        self.conv_layer4 = Conv2D(filters=48, kernel_size=3, padding='same', activation=tf.nn.relu,
                                  kernel_regularizer=reg, bias_regularizer=reg, activity_regularizer=reg)
        self.global_pooling = GlobalAvgPool2D()
        self.out = Dense(10, activation=tf.nn.softmax,
                         kernel_regularizer=reg, bias_regularizer=reg, activity_regularizer=reg)

    @tf.function
    def call(self, inputs):
        return self.out(self.global_pooling(
            self.conv_layer4(self.conv_layer3(self.pooling1(self.conv_layer2(self.conv_layer1(inputs)))))))


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

train_ds, test_ds = tfds.load('FashionMNIST', split=['train', 'test'], as_supervised=True)
train_dataset = train_ds.apply(prepare_mnist_data)
test_dataset = test_ds.apply(prepare_mnist_data)

num_epochs = 10
learning_rate = 0.01  # 0.1 is to high

model = FashionMnistConvModel()
cross_entropy_loss = tf.keras.losses.CategoricalCrossentropy()
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
