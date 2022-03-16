import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Layer
from RecommenderDecoder import RecommenderDecoder
from RecommenderEncoder import RecommenderEncoder
from RecipeEmbedding import RecipeEmbedding


class FoodRecommenderModel(Layer):  # tf.Module):

    def __init__(self):
        super(FoodRecommenderModel, self).__init__()
        self.loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        self.max_seq_len = 20
        self.num_ids = 100000
        self.embedding = RecipeEmbedding(id_embedding_size=32, num_ids=self.num_ids, ingredient_embedding_size=128,
                                         other_features_embedding_size=16, sequence_length=self.max_seq_len)
        self.encoder = RecommenderEncoder(self.embedding.emb_size())
        self.decoder = RecommenderDecoder(self.embedding.emb_size())
        self.flatten = Flatten()
        self.out = Dense(1)

    def call(self, inputs, training=False):
        recipes, tar_recipe = inputs
        recipes = self.pad(recipes)
        recipes = self.embedding(recipes, positional=True)
        tar_recipe = self.embedding(tar_recipe)
        recipes_encoded = self.encoder(recipes, training)
        x = self.decoder(tar_recipe, recipes_encoded, training)
        x = self.flatten(x)
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
