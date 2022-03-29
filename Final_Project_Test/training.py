from FoodRecommenderModelSequence import FoodRecommenderModelSequence
import tensorflow as tf
import keras
import pandas as pd
from DatasetPreprossesing import DatasetPreprossesing
from FoodRatingDataSet import FoodRatingDataset

NUM_ING = 8023
OTHER_FEATURES = 3
NUM_RECIPES = 161880

BATCHSIZE = 1

SEQ_LEN = 9



newnew = FoodRatingDataset()



#ids, ing, ofe = tf.keras.Input(shape=[20]), tf.keras.Input(shape=[20, 6000]), tf.keras.Input(shape=[20, 3])

rec = FoodRecommenderModelSequence(recipe_count=NUM_RECIPES, seq_len=SEQ_LEN)

x = rec(newnew.prepre())

print(x)