import tensorflow as tf
from tensorflow.keras.layers import Layer
from TransformerLayer import TransformerLayer


class RecommenderEncoder(Layer):

    def __init__(self, embedding_size, hidden_size):
        super(RecommenderEncoder, self).__init__()
        self.transformer = TransformerLayer(output_size=embedding_size, hidden_size=hidden_size)

    @tf.function
    def call(self, inputs, training=False):
        # TODO: Padding Mask?
        x = self.transformer(inputs, inputs, training)
        return x
