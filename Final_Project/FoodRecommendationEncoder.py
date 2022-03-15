import tensorflow as tf
from tensorflow.keras.layers import Layer
from Final_Project.RecipeEmbedding import RecipeEmbedding
from Final_Project.TransformerLayer import TransformerLayer


class FoodRecommendationEncoder(Layer):

    def __init__(self, embedding_sice):
        super(FoodRecommendationEncoder, self).__init__()
        self.embedding = RecipeEmbedding()
        self.transformer = TransformerLayer(embedding_sice, 4, 128)

    def call(self, inputs, training=False):
        x = self.embedding(inputs, training)
        return self.transformer(x, training)
