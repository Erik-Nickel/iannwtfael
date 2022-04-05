from tensorflow.keras.layers import Layer, Embedding, Add, Concatenate, Dense, Flatten
import tensorflow as tf


class RecipeEmbedding(Layer):

    def __init__(self, id_embedding_size, num_ids, ingredient_embedding_size, other_features_embedding_size,
                 sequence_length, output_size):
        super(RecipeEmbedding, self).__init__()

        self.output_size = output_size
        self.id_embedding_size = id_embedding_size
        self.ingredient_embedding_size = ingredient_embedding_size
        self.other_features_embedding_size = other_features_embedding_size
        self.sequence_length = sequence_length

        self.recipy_id_embedding = Embedding(num_ids, id_embedding_size)
        self.position_embedding = Embedding(self.sequence_length, output_size)
        self.add = Add()
        self.flatten = Flatten()
        self.ingredient_embedding = Dense(ingredient_embedding_size, activation=tf.nn.tanh)
        self.other_features_embedding = Dense(other_features_embedding_size, activation=tf.nn.tanh)
        self.concat = Concatenate(axis=-1)
        self.out = Dense(output_size, activation=tf.nn.tanh)

    @tf.function
    def call(self, inputs, positional=False):
        recipe_id, ing, other_features = inputs
        x_id = self.recipy_id_embedding(recipe_id)
        x_ing = self.ingredient_embedding(ing)
        # x_o = self.other_features_embedding(other_features) TODO: activate if normalized
        x = self.concat([x_id, x_ing])  # , x_o])
        x = self.out(x)
        if positional:
            positions = tf.range(start=0, limit=self.sequence_length)
            positions = tf.expand_dims(positions, axis=0)
            positions = tf.repeat(positions, tf.shape(x)[0], axis=0)
            embedded_positions = self.position_embedding(positions)
            x = self.add([x, embedded_positions])
        return x

    def emb_size(self):
        return self.output_size
