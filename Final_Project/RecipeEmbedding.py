from tensorflow.keras.layers import Layer, Embedding, Add
import tensorflow as tf


class RecipeEmbedding(Layer):

    def __init__(self, embedding_size, num_ids, sequence_length):
        super(RecipeEmbedding, self).__init__()
        self.sequence_length = sequence_length
        self.output_size = embedding_size
        self.recipy_id_embedding = Embedding(num_ids, embedding_size)
        self.position_embedding = Embedding(sequence_length, embedding_size)
        self.add = Add()

    def call(self, inputs):
        x = self.recipy_id_embedding(inputs)
        positions = tf.range(start=0, limit=self.sequence_length)
        positions = tf.expand_dims(positions, axis=0)
        positions = tf.repeat(positions, tf.shape(x)[0], axis=0)
        embedded_positions = self.position_embedding(positions)
        x = self.add([x, embedded_positions])
        return x

    def emb_size(self):
        return self.output_size
