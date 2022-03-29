from tensorflow.keras.layers import CategoryEncoding
import tensorflow as tf


class TestDataset:
    NUM_ING = 20
    OTHER_FEATURES = 3
    NUM_RECIPES = 161880

    __ids = [[1, 2, 3, 4, 5]]
    __ingr = [[[5, 1, 0, 15],
               [10, 16, 14],
               [3, 11, 13, 17, 15, 11],
               [1, 14, 6, 9],
               [16, 6, 18]]]
    __ohter = [[[1, 2, 3],
                [5, 8, 9],
                [6, 2, 10],
                [5, 8, 1],
                [7, 6, 9]]]
    __target = [99]

    def __init__(self):
        super(TestDataset, self).__init__()
        self.catenc = CategoryEncoding(num_tokens=self.NUM_ING, output_mode="multi_hot")

    def __call__(self, id, ing, ofe):
        return id, self.catenc(ing), ofe

    def data(self):
        return tf.data.Dataset.from_tensor_slices((self.__ids, self.ing_mht(), self.__ohter, self.__target))
        # return (None, None, None), None

    def ing_mht(self):
        return [[self.catenc(i) for i in self.__ingr[0]]]


ds = TestDataset()
print(ds.data())
