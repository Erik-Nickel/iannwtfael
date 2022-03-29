import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalMaxPool1D, Flatten
from RecommenderEncoder import RecommenderEncoder
from RecipeEmbedding import RecipeEmbedding


class FoodRecommenderModelSequence(tf.keras.Model):

    def __init__(self, recipe_count, seq_len):
        super(FoodRecommenderModelSequence, self).__init__()
        self.max_seq_len = seq_len
        self.num_ids = recipe_count
        self.embedding = RecipeEmbedding(id_embedding_size=32, num_ids=self.num_ids, ingredient_embedding_size=128,
                                         other_features_embedding_size=16, sequence_length=self.max_seq_len)
        self.encoder = RecommenderEncoder(self.embedding.emb_size())
        self.pooling = GlobalMaxPool1D()  # or flatten
        self.out = Dense(self.num_ids)

    def call(self, inputs, training=False):
        x = self.embedding(inputs, positional=True)
        x = self.encoder(x, training)
        x = self.pooling(x)  # or flatten
        # more layers?
        x = self.out(x)
        return x

    def pad(self, inp):
        s = self.max_seq_len
        return inp  # TODO

