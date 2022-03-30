import tensorflow as tf
from tensorflow.keras.layers import Dense, Layer, GlobalMaxPool1D, Flatten
from RecommenderEncoder import RecommenderEncoder
from RecipeEmbedding import RecipeEmbedding


class FoodRecommenderModelSequence(tf.keras.Model):

    def __init__(self, recipe_count, seq_len):
        super(FoodRecommenderModelSequence, self).__init__()
        #self.loss = tf.keras.losses.MeanSquaredError()
        #self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        self.max_seq_len = seq_len
        self.num_ids = recipe_count
        self.embedding = RecipeEmbedding(id_embedding_size=128*128, num_ids=self.num_ids, ingredient_embedding_size=128*128,
                                         other_features_embedding_size=128*128, sequence_length=self.max_seq_len)
        self.encoder = RecommenderEncoder(self.embedding.emb_size())
        self.pooling = GlobalMaxPool1D()  # or flatten
        self.out = Dense(self.num_ids)

    def call(self, inputs, training=False):
        x = self.embedding(inputs, positional=True)
        x = self.encoder(x, training)
        x = self.pooling(x)  # or flatten
        # more layers?
        x = self.out(x)
        return x

    def pad(self, inp):
        s = self.max_seq_len
        return inp  # TODO

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        print("DATA:")
        x, y = data
        print("X:", x)
        print("Y:", y)

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}


