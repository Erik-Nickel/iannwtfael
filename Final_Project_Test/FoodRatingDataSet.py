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
        self.dataset = tf.data.Dataset.from_generator(self.new.genData2,output_signature=((tf.TensorSpec(shape=(9), dtype=tf.int32),tf.TensorSpec(shape=(9,8023), dtype=tf.int32),tf.TensorSpec(shape=(9,3), dtype=tf.int32)),tf.TensorSpec(shape=(), dtype=tf.int32)))


      



    def data(self,btchsz = 32):
      
        return self.dataset.batch(btchsz).prefetch(64)#.shuffle(32)   # # # (None, None, None), None

    