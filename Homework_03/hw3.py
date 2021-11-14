import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense

train_ds, test_ds = tfds.load('genomics_ood', split=['train', 'test'], as_supervised=True)


def onehotify(tensor):
    vocab = {'A': '1', 'C': '2', 'G': '3', 'T': '0'}
    for key in vocab.keys():
        tensor = tf.strings.regex_replace(tensor, key, vocab[key])
    split = tf.strings.bytes_split(tensor)
    labels = tf.cast(tf.strings.to_number(split), tf.uint8)
    onehot = tf.one_hot(labels, 4)
    onehot = tf.reshape(onehot, (-1,))
    return onehot


def prepare_genomics_odd(genomics_ood):
    genomics_ood = genomics_ood.map(lambda point, label: (onehotify(point), tf.one_hot(label, 10)))
    genomics_ood = genomics_ood.shuffle(10000000)
    genomics_ood = genomics_ood.batch(80)
    genomics_ood = genomics_ood.prefetch(100000)
    return genomics_ood


train_dataset = train_ds.apply(prepare_genomics_odd)
test_dataset = test_ds.apply(prepare_genomics_odd)


class BacteriaGeneModel(tf.keras.Model):

    def __init__(self):
        super(BacteriaGeneModel, self).__init__()
        self.dense1 = Dense(256, activation=tf.nn.sigmoid)
        self.dense2 = Dense(256, activation=tf.nn.sigmoid)
        self.out = Dense(10, activation=tf.nn.softmax)

    @tf.function
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.out(x)
        return x


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

train_dataset = train_dataset.take(100000)
test_dataset = test_dataset.take(1000)

num_epochs = 10
learning_rate = 0.1

model = BacteriaGeneModel()
cross_entropy_loss = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.SGD(learning_rate)

train_losses = []
test_losses = []
test_accuracies = []

test_loss, test_accuracy = test(model, test_dataset, cross_entropy_loss)
test_losses.append(test_loss)
test_accuracies.append(test_accuracy)

train_loss, _ = test(model, train_dataset, cross_entropy_loss)
train_losses.append(train_loss)

for epoch in range(num_epochs):
    epoch_loss_agg = []
    for input, target in train_dataset:
        train_loss = train_step(model, input, target, cross_entropy_loss, optimizer)
        epoch_loss_agg.append(train_loss)
    train_losses.append(tf.reduce_mean(epoch_loss_agg))

    test_loss, test_accuracy = test(model, test_dataset, cross_entropy_loss)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)

plt.plot(train_losses, label="training")
plt.plot(test_losses, label="test")
plt.plot(test_accuracies, label="test accuracy")
plt.xlabel("Training steps")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
