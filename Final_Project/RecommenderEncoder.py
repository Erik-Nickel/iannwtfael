import tensorflow as tf
from tensorflow.keras.layers import Layer
from TransformerLayer import TransformerLayer


class RecommenderEncoder(Layer):

    def __init__(self, embedding_size, hidden_size, num_layers=1):
        super(RecommenderEncoder, self).__init__()
        self.t_layers = [TransformerLayer(output_size=embedding_size, hidden_size=hidden_size) for _ in
                         range(num_layers)]

    def call(self, inputs, training=False):
        x = inputs
        for t in self.t_layers:
            x = t(x, x, training)
        return x
