import tensorflow as tf
from tensorflow.keras.layers import Dense
from Final_Project.FoodRecommendationDecoder import FoodRecommendationDecoder
from Final_Project.FoodRecommendationEncoder import FoodRecommendationEncoder


class FoodRecommenderModel(tf.Module):

    def __init__(self, num_recipes):
        super(FoodRecommenderModel, self).__init__()
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        self.encoder = FoodRecommendationEncoder()
        self.decoder = FoodRecommendationDecoder()
        self.out = Dense(num_recipes)  # or tf.nn.softmax and CategoricalCrossentropy

    def call(self, inputs, training=False):
        inp, tar = inputs
        x = self.encoder(inp, training)
        x = self.decoder(x, tar, training)
        x = self.out(x)
        return x

    def recommend(self, liked_recipes, count=10):
        recommended_rec = ['start_token']  # TODO: start Token
        for i in range(count):
            pred = self.call((self.pad(liked_recipes), self.pad(recommended_rec)))
            index_of_max = pred  # TODO
            recommended_rec.append(index_of_max)
            liked_recipes.append(index_of_max)
        return recommended_rec[1:]

    def pad(self, inp):
        return inp  # TODO
