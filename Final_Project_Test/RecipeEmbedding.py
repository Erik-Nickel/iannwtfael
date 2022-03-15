from tensorflow.keras.layers import Layer, Embedding, Add, Concatenate, Dense, Flatten
import tensorflow as tf


class RecipeEmbedding(Layer):

    def __init__(self, id_embedding_size, num_ids, ingredient_embedding_size,
                 other_features_embedding_size):
        super(RecipeEmbedding, self).__init__()

        self.id_embedding_size = id_embedding_size
        self.ingredient_embedding_size = ingredient_embedding_size
        self.other_features_embedding_size = other_features_embedding_size

        # self.positional_embedding = Embedding(sequence_len, embedding_sice)
        self.recipy_id_embedding = Embedding(num_ids, id_embedding_size)
        self.flatten = Flatten()
        self.ingredient_embedding = Dense(ingredient_embedding_size)
        self.other_features_embedding = Dense(other_features_embedding_size)
        self.concat = Concatenate()

    def call(self, inputs, training=False):
        recipe_id, ing, other_features = inputs
        # tf.reduce_max(tf.one_hot(labels, num_classes, dtype=tf.int32), axis=0)
        # x_id = self.flatten(x_id)
        x_id = self.recipy_id_embedding(recipe_id)
        x_ing = self.ingredient_embedding(ing)
        x_o = self.other_features_embedding(other_features)
        return self.concat([x_id, x_ing, x_o])

    def emb_size(self):
        return self.id_embedding_size + self.ingredient_embedding_size + self.other_features_embedding_size
