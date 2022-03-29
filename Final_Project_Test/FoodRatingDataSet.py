import tensorflow as tf
from tensorflow.keras.layers import CategoryEncoding
import numpy as np
from DatasetPreprossesing import DatasetPreprossesing




class FoodRatingDataset:
    NUM_ING = 8023
    OTHER_FEATURES = 3
    NUM_RECIPES = 161880

    def __init__(self):
        super(FoodRatingDataset, self).__init__()
        self.new = DatasetPreprossesing()
        self.new.preprocessing()

        self.dataset = tf.data.Dataset.from_generator(self.new.genData)

        

    def prepre(self,batchsize = 1):
        self.dataset = self.dataset.batch(batchsize)
        return(self.dataset)

    def data(self):
        return (None, None, None), None

    # tf.reduce_max(tf.one_hot(labels, num_classes, dtype=tf.int32), axis=0)
