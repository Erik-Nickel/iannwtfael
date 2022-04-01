import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from RecommenderEncoder import RecommenderEncoder
from RecipeEmbedding import RecipeEmbedding


class FoodRecommenderModelSequence(tf.keras.Model):

    def __init__(self, recipe_count, seq_len):
        super(FoodRecommenderModelSequence, self).__init__()
        self.max_seq_len = seq_len
        self.num_ids = recipe_count
        self.embedding = RecipeEmbedding(id_embedding_size=64, num_ids=self.num_ids, ingredient_embedding_size=128,
                                         other_features_embedding_size=32, sequence_length=self.max_seq_len,
                                         output_size=128)
        self.encoder = RecommenderEncoder(embedding_size=self.embedding.emb_size(), hidden_size=256)
        self.accumulate = Flatten()  # or GlobalPool1D
        self.dene1 = Dense(512, activation=tf.nn.leaky_relu)
        self.dene2 = Dense(256, activation=tf.nn.leaky_relu)
        self.dene3 = Dense(128, activation=tf.nn.leaky_relu)
        self.out = Dense(self.num_ids)  # softmax activation if not loss from_logits=True

    @tf.function
    def call(self, inputs, training=False):
        x = self.embedding(inputs, positional=True)
        x = self.encoder(x, training)
        x = self.accumulate(x)
        x = self.dene1(x)
        x = self.dene2(x)
        x = self.dene3(x)
        x = self.out(x)
        return x

    def pad(self, inp):
        s = self.max_seq_len
        return inp  # TODO
