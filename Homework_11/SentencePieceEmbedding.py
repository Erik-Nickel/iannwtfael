from tensorflow.keras.layers import Layer, Embedding, Add
import tensorflow as tf


class SentencePieceEmbedding(Layer):

    def __init__(self, vocabulary_sice, embedding_sice, sequence_len):
        super(SentencePieceEmbedding, self).__init__()
        self.positional_embedding = Embedding(sequence_len, embedding_sice)
        self.token_embedding = Embedding(vocabulary_sice, embedding_sice)
        self.add_layer = Add()

    def call(self, inputs, *args, **kwargs):
        (batch_size, seq_len) = inputs.shape()
        positions = tf.range(seq_len)  # TODO: batch_size?
        x_p = self.positional_embedding(positions)
        x_t = self.token_embedding(inputs)
        out = self.add_layer([x_p, x_t])
        return out
