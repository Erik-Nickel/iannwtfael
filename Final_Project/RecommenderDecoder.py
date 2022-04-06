from tensorflow.keras.layers import Layer, LayerNormalization, Add, Dropout, MultiHeadAttention
from TransformerLayer import TransformerLayer


class RecommenderDecoder(Layer):

    def __init__(self, embedding_size):
        super(RecommenderDecoder, self).__init__()

        attention_heads = 4
        dropout_rate = 0.1
        norm_epsilon = 1e-6

        self.attention = MultiHeadAttention(attention_heads, embedding_size)
        self.dropout = Dropout(dropout_rate)
        self.add = Add()
        self.norm = LayerNormalization(epsilon=norm_epsilon)
        self.enc_trans = TransformerLayer(embedding_size, 128, attention_heads, dropout_rate, norm_epsilon)

    def call(self, inputs, enc_output, training=False):
        # input transf
        x = self.attention(inputs, inputs)
        x = self.dropout(x, training)
        x = self.add([x, inputs])
        x = self.norm(x)
        # encoding transf
        x = self.enc_trans(inputs, enc_output, training)
        return x
