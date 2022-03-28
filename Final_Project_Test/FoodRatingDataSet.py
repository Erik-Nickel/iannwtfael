from tensorflow.keras.layers import CategoryEncoding


class FoodRatingDataset:

    def __init__(self):
        super(FoodRatingDataset, self).__init__()
        self.catenc = CategoryEncoding(num_tokens=8023, output_mode="multi_hot")

    def __call__(self,id,ing,ofe):
        return id,self.catenc(ing),ofe




# tf.reduce_max(tf.one_hot(labels, num_classes, dtype=tf.int32), axis=0)
