class FoodRatingDataset:
    NUM_ING = 8023
    OTHER_FEATURES = 3
    NUM_RECIPES = 161880

    def __init__(self, seq_len=10, batch_size=32):
        pass

    def data(self):
        return (None, None, None), None

    # tf.reduce_max(tf.one_hot(labels, num_classes, dtype=tf.int32), axis=0)
