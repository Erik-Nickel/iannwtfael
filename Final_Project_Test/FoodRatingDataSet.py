import tensorflow as tf
from tensorflow.keras.layers import CategoryEncoding
import numpy as np
from DatasetPreprossesing import DatasetPreprossesing





class FoodRatingDataset:
    NUM_ING = 8023
    OTHER_FEATURES = 3
    NUM_RECIPES = 178265

    


    def __init__(self, seq_len):
        super(FoodRatingDataset, self).__init__()
        self.new = DatasetPreprossesing(seq_len + 1)
        self.new.preprocessing()
        self.data_train = tf.data.Dataset.from_generator(self.new.gen_data_train,output_signature=((tf.TensorSpec(shape=(seq_len), dtype=tf.int32),tf.TensorSpec(shape=(seq_len,8023), dtype=tf.int32),tf.TensorSpec(shape=(seq_len,3), dtype=tf.int32)),tf.TensorSpec(shape=(), dtype=tf.int32)))
        self.data_val = tf.data.Dataset.from_generator(self.new.gen_data_val,output_signature=((tf.TensorSpec(shape=(seq_len), dtype=tf.int32),tf.TensorSpec(shape=(seq_len,8023), dtype=tf.int32),tf.TensorSpec(shape=(seq_len,3), dtype=tf.int32)),tf.TensorSpec(shape=(), dtype=tf.int32)))




    def data(self,btchsz = 32):
      
        return self.dataset.batch(btchsz).prefetch(btchsz*6) #.shuffle(btchsz*2) # (None, None, None), None

    