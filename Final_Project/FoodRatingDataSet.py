import tensorflow as tf
from DatasetPreprocessing import DatasetPreprocessing


class FoodRatingDataset:
    NUM_ING = 8023
    OTHER_FEATURES = 3
    NUM_RECIPES = 178265

    def __init__(self, seq_len=9):
        super(FoodRatingDataset, self).__init__()

        self.preprocessing = DatasetPreprocessing(seq_len)
        self.preprocessing.preprocesses(dataset_chunk=0.33)
        self.seq_len = seq_len
        self.data_train = self.gen_dataset(self.preprocessing.gen_data_train)
        self.data_val = self.gen_dataset(self.preprocessing.gen_data_val)

    def gen_dataset(self, generator):
        return tf.data.Dataset.from_generator(generator, output_signature=(
        tf.TensorSpec(shape=self.seq_len, dtype=tf.int32), tf.TensorSpec(shape=(), dtype=tf.int32)))

    def data(self, batch_size=256):
        train = self.data_train.batch(batch_size)
        val = self.data_val.batch(batch_size)
        return train, val
