from tensorflow.keras.layers import Layer, Dense, MultiHeadAttention, LayerNormalization, Dropout, Add
import tensorflow as tf


class TransformerLayer(Layer):

    def __init__(self, output_size, hidden_size, attention_heads=4, dropout_rate=0.2, norm_epsilon=1e-6):
        super(TransformerLayer, self).__init__()
        self.attention = MultiHeadAttention(attention_heads, output_size)
        self.norm1 = LayerNormalization(epsilon=norm_epsilon)
        self.norm2 = LayerNormalization(epsilon=norm_epsilon)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)
        self.dense1 = Dense(hidden_size, activation=tf.nn.leaky_relu)
        self.dense2 = Dense(output_size, activation=tf.nn.leaky_relu)
        self.add1 = Add()
        self.add2 = Add()

    def call(self, q, v, training):
        x = self.attention(q, v)
        x = self.dropout1(x, training)
        x = self.add1([x, q])
        ln_out = self.norm1(x)
        x = self.dense1(ln_out)
        x = self.dense2(x)
        x = self.dropout2(x, training)
        x = self.add2([x, ln_out])
        x = self.norm2(x)
        return x
