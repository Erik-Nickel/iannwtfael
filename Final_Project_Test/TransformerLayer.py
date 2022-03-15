from tensorflow.keras.layers import Layer, Dense, MultiHeadAttention, LayerNormalization, Dropout, Add
import tensorflow as tf


class TransformerLayer(Layer):

    def __init__(self, output_size, hidden_size, attention_heads=4, dropout_rate=0.1, norm_epsilon=1e-6):
        super(TransformerLayer, self).__init__()
        self.attention = MultiHeadAttention(attention_heads, output_size)
        self.norm1 = LayerNormalization(epsilon=norm_epsilon)
        self.norm2 = LayerNormalization(epsilon=norm_epsilon)
        self.dropout = Dropout(dropout_rate)
        self.dense1 = Dense(hidden_size, activation=tf.nn.relu)
        self.dense2 = Dense(output_size)
        self.add1 = Add()
        self.add2 = Add()

    def call(self, q, v, training):
        x = self.attention(q, v)
        x = self.dropout(x, training)
        x = self.add1([x, q])
        ln_out = self.norm1(x)
        x = self.dense1(ln_out)
        x = self.dense2(x)
        x = self.add2([x, ln_out])
        x = self.norm2(x)
        return x
