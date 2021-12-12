import tensorflow as tf
from keras.layers import Dense, Activation, Concatenate, LSTM


class LstmLayer(tf.keras.layers.Layer):
    def __init__(self, LSTM_Cell, return_sequences=False):
        super(LstmLayer, self).__init__()
        self.return_sequences = return_sequences
        self.cell = LSTM_Cell

    def call(self, data, training=False):
        length = data.shape[1]
        h_state = tf.zeros((data.shape[0], self.cell.units), tf.float32)
        c_state = tf.ones((data.shape[0], self.cell.units), tf.float32)
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


input_shape = (2, 6, 16)
hidden_dim = 3
input_sequence = tf.random.uniform(shape=input_shape)
simple_RNN_cell = LstmCell(hidden_dim)
RNN = LstmLayer(simple_RNN_cell, return_sequences=True)
output = RNN(input_sequence)
print(f"The output of the RNN, which is the last hidden state is \n{output}\n\n")
