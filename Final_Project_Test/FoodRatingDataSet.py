from tensorflow.keras.layers import CategoryEncoding


class FoodRatingDataset:
    NUM_ING = 8023
    OTHER_FEATURES = 3
    NUM_RECIPES = 161880

    def __init__(self):
        super(FoodRatingDataset, self).__init__()
        self.catenc = CategoryEncoding(num_tokens=8023, output_mode="multi_hot")

    def __call__(self,id,ing,ofe):
        return id,self.catenc(ing),ofe



    def data(self):
        return (None, None, None), None

    # tf.reduce_max(tf.one_hot(labels, num_classes, dtype=tf.int32), axis=0)
