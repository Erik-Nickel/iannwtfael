from FoodRecommenderModelSequence import FoodRecommenderModelSequence
import tensorflow as tf
import keras
import pandas as pd
from DatasetPreprossesing import DatasetPreprossesing
from FoodRatingDataSet import FoodRatingDataset

new = DatasetPreprossesing()
new.preprocessing()

newnew = FoodRatingDataset()

for i in new.genData():
    print(i.ing_ids)
    lala = newnew(i.recipe_id,i.ing_ids,i.recipe_features)



#ids, ing, ofe = tf.keras.Input(shape=[20]), tf.keras.Input(shape=[20, 6000]), tf.keras.Input(shape=[20, 3])

x = FoodRecommenderModelSequence()(lala)

mod = keras.Model(lala, x)

mod.summary()