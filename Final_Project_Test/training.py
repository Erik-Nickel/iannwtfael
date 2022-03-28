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
    print(i.recipe_id)
    print(i.ing_ids)
    print(i.recipe_id)
    lala = newnew(id = i.recipe_id.to_numpy(), ing = i.ing_ids.to_numpy(), ofe = i.recipe_features.to_numpy())
    break


#ids, ing, ofe = tf.keras.Input(shape=[20]), tf.keras.Input(shape=[20, 6000]), tf.keras.Input(shape=[20, 3])

x = FoodRecommenderModelSequence()(lala)

mod = keras.Model(lala, x)

mod.summary()