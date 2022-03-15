from tensorflow.keras.layers import Layer, Embedding, Add, Concatenate
import tensorflow as tf


class RecipeEmbedding(Layer):

    def __init__(self, sequence_len, embedding_sice):
        super(RecipeEmbedding, self).__init__()
        self.positional_embedding = Embedding(sequence_len, embedding_sice)
        self.recipy_id_embedding = Embedding()
        self.add = Add()
        self.concat = Concatenate()

    def call(self, inputs, training=False):
        (batch_size, seq_len) = inputs.shape()
        positions = tf.range(seq_len)  # TODO: batch_size?
        x_p = self.positional_embedding(positions)
        x_t = self.recipy_id_embedding(inputs)
        out = self.add_layer([x_p, x_t])
        return out
