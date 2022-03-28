import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Layer, Concatenate, GlobalMaxPool1D
from RecommenderEncoder import RecommenderEncoder
from RecipeEmbedding import RecipeEmbedding


class FoodRecommenderModelSelfAttention(Layer):  # tf.Module):

    def __init__(self):
        super(FoodRecommenderModelSelfAttention, self).__init__()
        self.loss = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        self.max_seq_len = 20
        self.num_ids = 100000
        self.embedding = RecipeEmbedding(id_embedding_size=32, num_ids=self.num_ids, ingredient_embedding_size=128,
                                         other_features_embedding_size=16, sequence_length=self.max_seq_len)
        self.encoder = RecommenderEncoder(self.embedding.emb_size())
        # self.flatten1 = Flatten()
        self.flatten2 = Flatten()
        self.pooling = GlobalMaxPool1D()
        self.concat = Concatenate()
        self.out = Dense(1)

    def call(self, inputs, training=False):
        recipes, tar_recipe = inputs
        recipes = self.embedding(recipes)
        tar_recipe = self.embedding(tar_recipe)
        recipes_encoded = self.encoder(recipes, training)
        # recipes_encoded = self.flatten1(recipes_encoded)
        recipes_encoded = self.pooling(recipes_encoded)
        tar_recipe = self.flatten2(tar_recipe)
        x = self.concat([tar_recipe, recipes_encoded])
        # more layers?
        x = self.out(x)
        return x

    def recommend(self, liked_recipes, recipes_selection, top_x=10):
        rated = []
        liked_recipes_padded = self.pad(liked_recipes)
        for r in recipes_selection:
            rating = self.call((liked_recipes_padded, r))
            rated.append((rating, r))
        return rated  # return top_x ratings

    def pad(self, inp):
        s = self.max_seq_len
        return inp  # TODO
