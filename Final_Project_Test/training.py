from FoodRecommenderModelSequence import FoodRecommenderModelSequence
import tensorflow as tf
import keras
import pandas as pd
from DatasetPreprossesing import DatasetPreprossesing
from FoodRatingDataSet import FoodRatingDataset

#from TestDataSet import TestDataSet


NUM_ING = 8023
OTHER_FEATURES = 3
NUM_RECIPES = 178265

BATCHSIZE = 1

SEQ_LEN = 9

#ds = TestDataSet()
#rec = FoodRecommenderModelSequence(recipe_count=NUM_RECIPES, seq_len=SEQ_LEN)
#for (d, v) in ds.data():
#    x = rec(d)
#    print(x)

print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~newnew~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
newnew = FoodRatingDataset()
#print(newnew.data())
data = newnew.data()


print("DATA: ", data)
#ids, ing, ofe = tf.keras.Input(shape=[20]), tf.keras.Input(shape=[20, 6000]), tf.keras.Input(shape=[20, 3])

rec = FoodRecommenderModelSequence(recipe_count=NUM_RECIPES, seq_len=SEQ_LEN)
rec.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), metrics=['accuracy'])
#x = rec(data)
rec.fit(data, epochs=10)
#print(x)