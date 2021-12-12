import tensorflow as tf
from keras.layers import Dense, Activation, Concatenate, LSTM
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np


def prepare_mnist_data(ds):
    return ds.batch(8).shuffle(1024).batch(32).prefetch(32)  # TODO finish!!!!


class LstmLayer(tf.keras.layers.Layer):
    def __init__(self, LSTM_Cell, return_sequences=False):
        super(LstmLayer, self).__init__()
        self.return_sequences = return_sequences
        self.cell = LSTM_Cell

    def call(self, data, training=False):
        length = data.shape[1]
        h_state = tf.zeros((data.shape[0], self.cell.units), tf.float32)
        c_state = tf.zeros((data.shape[0], self.cell.units), tf.float32)
        hidden_states = tf.TensorArray(dtype=tf.float32, size=length)
        for t in tf.range(length):
            input_t = data[:, t, :]
            h_state, c_state = self.cell(input_t, h_state, c_state, training)
            if self.return_sequences:
                hidden_states.write(t, h_state)
        if self.return_sequences:
            # transpose the sequence of hidden_states from TensorArray accordingly
            # (batch and time dimensions are otherwise switched after .stack())
            outputs = tf.transpose(hidden_states.stack(), [1, 0, 2])
        else:
            outputs = h_state
        return outputs


class LstmCell(tf.keras.layers.Layer):
    def __init__(self, units, kernel_regularizer=None):
        super(LstmCell, self).__init__()
        self.units = units
        self.forget_gate = Dense(units, activation='sigmoid')
        self.input_gate = Dense(units, activation='sigmoid')
        self.cell_state_candidates = Dense(units, activation='tanh')
        self.output_gate = Dense(units, activation='sigmoid')
        self.tanh_layer = Activation(activation='tanh')
        self.concat_layer = Concatenate(axis=-1)

    def call(self, x_t, h_t, c_t, training=False):
        x_t_h_t = self.concat_layer([h_t, x_t])
        f = self.forget_gate(x_t_h_t)
        i = self.input_gate(x_t_h_t)
        c_h = self.cell_state_candidates(x_t_h_t)
        o = self.output_gate(x_t_h_t)
        c = (f * c_t) + (i * c_h)
        h = o * self.tanh_layer(c)
        return h, c


class LstmModel(tf.keras.Model):

    def __init__(self):
        super(LstmModel, self).__init__()
        units = 32
        self.embedding = tf.keras.layers.Dense(16, activation='sigmoid')
        # self.LSTM = LSTM(units=units, return_sequences=False)
        self.LSTM = LstmLayer(LstmCell(units), return_sequences=False)
        self.out = Dense(1, activation='sigmoid')

    def call(self, x):
        x = self.embedding(x)
        x = self.LSTM(x)
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
        sample_test_accuracy = np.round(target, 0) == np.round(prediction, 0)
        sample_test_accuracy = np.mean(sample_test_accuracy)
        test_loss_aggregator.append(sample_test_loss.numpy())
        test_accuracy_aggregator.append(np.mean(sample_test_accuracy))

    test_loss = tf.reduce_mean(test_loss_aggregator)
    test_accuracy = tf.reduce_mean(test_accuracy_aggregator)

    return test_loss, test_accuracy


tf.keras.backend.clear_session()

train_ds, test_ds = []  # TODO!!!!!
train_dataset = train_ds.apply(prepare_mnist_data)
test_dataset = test_ds.apply(prepare_mnist_data)

num_epochs = 10
learning_rate = 0.1  # TODO parameter tuning!!!!

model = LstmModel()
loss = tf.keras.losses.MeanSquaredError()  # TODO: Loss Type ????
optimizer = tf.keras.optimizers.Adam(learning_rate)

train_losses = []
test_losses = []
test_accuracies = []

test_loss, test_accuracy = test(model, test_dataset, loss)
test_losses.append(test_loss)
test_accuracies.append(test_accuracy)

train_loss, _ = test(model, train_dataset, loss)
train_losses.append(train_loss)

for epoch in range(num_epochs):
    print(f'Epoch: {str(epoch)} starting with accuracy {test_accuracies[-1]}')
    epoch_loss_agg = []
    for input, target in train_dataset:
        train_loss = train_step(model, input, target, loss, optimizer)
        epoch_loss_agg.append(train_loss)
    train_losses.append(tf.reduce_mean(epoch_loss_agg))
    test_loss, test_accuracy = test(model, test_dataset, loss)
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
