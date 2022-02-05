from tensorflow.keras.layers import Layer, Dense, MultiHeadAttention, LayerNormalization, Dropout, Add
import tensorflow as tf


class Transformer(Layer):

    def __init__(self, embedding_size):
        self.attention = MultiHeadAttention(4, embedding_size)
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.norm2 = LayerNormalization(epsilon=1e-6)
        self.dropout = Dropout(0.1)
        self.dense1 = Dense(256, activation=tf.nn.relu)  # 32-256
        self.dense2 = Dense(embedding_size)
        self.add1 = Add()
        self.add2 = Add()

    def call(self, inputs, training):
        x = self.attention(inputs, inputs, inputs)
        x = self.dropout(x, training)
        x = self.add([x, inputs])
        ln_out = self.norm1(x)
        x = self.dense1(ln_out)
        x = self.dense2(x)
        x = self.add2([x, ln_out])
        x = self.norm2(x)
        return x
