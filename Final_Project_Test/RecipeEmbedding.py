from tensorflow.keras.layers import Layer, Embedding, Add, Concatenate, Dense, Flatten
import tensorflow as tf


class RecipeEmbedding(Layer):

    def __init__(self, id_embedding_size, num_ids, ingredient_embedding_size,
                 other_features_embedding_size, sequence_length, output_size=128):
        super(RecipeEmbedding, self).__init__()

        self.output_size = output_size
        self.id_embedding_size = id_embedding_size
        self.ingredient_embedding_size = ingredient_embedding_size
        self.other_features_embedding_size = other_features_embedding_size
        self.sequence_length = sequence_length

        self.recipy_id_embedding = Embedding(num_ids, id_embedding_size)
        self.position_embedding = Embedding(self.sequence_length, id_embedding_size)
        self.add = Add()

        self.flatten = Flatten()
        self.ingredient_embedding = Dense(ingredient_embedding_size)
        self.other_features_embedding = Dense(other_features_embedding_size)
        self.concat = Concatenate(axis=-1)
        self.out = Dense(output_size)
        # TODO: output functions for dense

    def call(self, inputs, training=False, positional=False):
        recipe_id, ing, other_features = inputs
        x_id = self.recipy_id_embedding(recipe_id)
        if positional:
            positions = tf.range(start=0, limit=self.sequence_length)
            embedded_positions = self.position_embedding(positions)
            embedded_positions = tf.expand_dims(embedded_positions, axis=0)  # TODO: just for batch size 1
            print("id_pre:", x_id)
            print("POS_emb:", embedded_positions)
            x_id = self.add([x_id, embedded_positions])
        x_ing = self.ingredient_embedding(ing)
        x_o = self.other_features_embedding(other_features)
        x = self.concat([x_id, x_ing, x_o])
        x = self.out(x)
        return x

    def emb_size(self):
        return self.output_size
