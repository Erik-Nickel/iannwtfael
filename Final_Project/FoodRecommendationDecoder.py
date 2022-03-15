import tensorflow as tf
from tensorflow.keras.layers import Layer, LayerNormalization, Add, Dropout, MultiHeadAttention
from Final_Project.TransformerLayer import TransformerLayer


class FoodRecommendationDecoder(Layer):

    def __init__(self):
        super(FoodRecommendationDecoder, self).__init__()
        self.attention = MultiHeadAttention()
        self.dropout = Dropout()
        self.add = Add()
        self.norm = LayerNormalization()
        self.enc_trans = TransformerLayer()

    def call(self, inputs, enc_output, training=False):
        # input transf
        x = self.attention(inputs, inputs)
        x = self.dropout(x, training)
        x = self.add([x, inputs])
        x = self.norm1(x)
        # encoding transf
        x = self.enc_trans(enc_output, inputs, training)
        return x
