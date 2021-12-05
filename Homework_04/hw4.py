import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense

'''
In the following script you see the import and trining of the Wine data set
'''

df = pd.read_csv('winequality-red.csv', sep=";")
print(df.head())
print(df.describe())
print(df.max() - df.min())

threshold = df.quality.median()
target = df.pop('quality')

print("target:", target)

df_train, df_validation, df_test = np.split(df.sample(frac=1, random_state=42), [int(.7 * len(df)), int(.9 * len(df))])
target_train, target_validation, target_test = np.split(target.sample(frac=1, random_state=42),
                                                        [int(.7 * len(df)), int(.9 * len(df))])

print("df_train: ", df_train, ": df_train", )
train_dataset = tf.data.Dataset.from_tensor_slices((df_train.values, target_train.values))
validation_dataset = tf.data.Dataset.from_tensor_slices((df_validation.values, target_validation.values))
test_dataset = tf.data.Dataset.from_tensor_slices((df_test.values, target_test.values))


def make_binary(target):
    return target > threshold


def prepare_wine(wine):
    wine = wine.map(lambda point, label: (point, make_binary(label)))
    wine = wine.shuffle(100000)
    wine = wine.batch(8)  # mini-batches
    return wine


train_dataset = train_dataset.apply(prepare_wine)
validation_dataset = validation_dataset.apply(prepare_wine)
test_dataset = test_dataset.apply(prepare_wine)

for elem, label in train_dataset.take(1):
    print("elem,label")
    tf.print(elem, label)


class WineModel(tf.keras.Model):

    def __init__(self):
        super(WineModel, self).__init__()
        self.dense1 = Dense(16, kernel_regularizer='l2', bias_regularizer='l2', activity_regularizer='l2',
                            activation=tf.nn.sigmoid)  # L2 Regulization
        self.dense2 = Dense(16, kernel_regularizer='l2', bias_regularizer='l2', activity_regularizer='l2',
                            activation=tf.nn.sigmoid)
        self.out = Dense(1, kernel_regularizer='l2', bias_regularizer='l2', activity_regularizer='l2',
                         activation=tf.nn.sigmoid)

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
        # sample_test_accuracy = np.argmax(target, axis=1) == np.argmax(prediction, axis=1)
        sample_test_accuracy = np.round(target, 0) == np.round(prediction, 0)
        sample_test_accuracy = np.mean(sample_test_accuracy)
        test_loss_aggregator.append(sample_test_loss.numpy())
        test_accuracy_aggregator.append(np.mean(sample_test_accuracy))

    test_loss = tf.reduce_mean(test_loss_aggregator)
    test_accuracy = tf.reduce_mean(test_accuracy_aggregator)

    return test_loss, test_accuracy


tf.keras.backend.clear_session()

# train_ds = train_ds.take()
# test_ds = test_ds.take()

num_epochs = 10
learning_rate = 0.1

model = WineModel()
cross_entropy_loss = tf.keras.losses.BinaryCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate)  # Adam optimizer

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

    test_loss, test_accuracy = test(model, validation_dataset, cross_entropy_loss)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)

test_loss, test_accuracy = test(model, test_dataset, cross_entropy_loss)
test_losses.append(test_loss)
test_accuracies.append(test_accuracy)

plt.plot(train_losses, label="training")
plt.plot(test_losses, label="test/validation")
plt.plot(test_accuracies, label="test/validation accuracy")
plt.xlabel("Training steps")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
