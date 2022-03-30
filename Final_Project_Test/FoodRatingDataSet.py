import tensorflow as tf
from tensorflow.keras.layers import CategoryEncoding
import numpy as np
from DatasetPreprossesing import DatasetPreprossesing





class FoodRatingDataset:
    NUM_ING = 8023
    OTHER_FEATURES = 3
    NUM_RECIPES = 178265

    


    def __init__(self):
        super(FoodRatingDataset, self).__init__()
        self.new = DatasetPreprossesing()
        self.new.preprocessing()

        self.dataset = tf.data.Dataset.from_generator(self.new.genData,output_signature=((tf.TensorSpec(shape=(9), dtype=tf.int32),tf.TensorSpec(shape=(9,8023), dtype=tf.int32),tf.TensorSpec(shape=(9,3), dtype=tf.int32)),tf.TensorSpec(shape=(), dtype=tf.int32)))
        #print(dir(self.dataset))
        #print(self.dataset)
        

    def dataPipeline(self,batchsize = 1):
        self.dataset = self.dataset.batch(batchsize)
        return(self.dataset)

    def data(self):
       # print(self.dataset)
        #print(self.dataset.batch(1)) # (None, None, None), None
        return self.dataset.batch(10) # (None, None, None), None

    # tf.reduce_max(tf.one_hot(labels, num_classes, dtype=tf.int32), axis=0)
