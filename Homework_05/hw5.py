import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import GlobalAvgPool2D
import matplotlib.pyplot as plt

train_ds, test_ds = tfds.load('FashionMNIST', split=['train', 'test'], as_supervised=True)

def prepare_FashionMNIST_data(FashionMNIST):
    # convert data from uint8 to float32
    FashionMNIST = FashionMNIST.map(lambda img, target: (tf.cast(img, tf.float32), target))
    # sloppy input normalization, just bringing image values from range [0, 255] to [-1, 1]
    FashionMNIST = FashionMNIST.map(lambda img, target: ((img / 128.) - 1., target))
    # create one-hot targets
    FashionMNIST = FashionMNIST.map(lambda img, target: (img, tf.one_hot(target, depth=10)))
    # cache this progress in memory, as there is no need to redo it; it is deterministic after all
    FashionMNIST = FashionMNIST.cache()
    # shuffle, batch, prefetch
    FashionMNIST = FashionMNIST.shuffle(1000)
    FashionMNIST = FashionMNIST.batch(32)
    FashionMNIST = FashionMNIST.prefetch(20)
    # return preprocessed dataset
    return FashionMNIST




class FashionMNISTConvModel(tf.keras.Model):

    def __init__(self):
        super(FashionMNISTConvModel, self).__init__()
        self.conv_layer1 = Conv2D(filters=24, kernel_size=3, padding='same', activation=tf.nn.relu)
        self.conv_layer2 = Conv2D(filters=24, kernel_size=3, padding='same', activation=tf.nn.relu)
        self.pooling1 = MaxPool2D(pool_size=2, strides=2)
        self.conv_layer3 = Conv2D(filters=48, kernel_size=3, padding='same', activation=tf.nn.relu)
        self.conv_layer4 = Conv2D(filters=48, kernel_size=3, padding='same', activation=tf.nn.relu)
        self.global_pooling = GlobalAvgPool2D()
        self.dense1 = Dense(16,kernel_regularizer='l2',bias_regularizer = 'l2',activity_regularizer= 'l2', activation=tf.nn.sigmoid)
        self.out = Dense(10, activation=tf.nn.softmax)

    @tf.function
    def call(self, inputs):
        return self.out(self.dense1(self.global_pooling(
            self.conv_layer4(self.conv_layer3(self.pooling1(self.conv_layer2(self.conv_layer1(inputs))))))))

@tf.function
def train_step(model, input, target, loss_function, optimizer):
    # loss_object and optimizer_object are instances of respective tensorflow classes
    with tf.GradientTape() as tape:
        prediction = model(input)
        loss = loss_function(target, prediction)
        gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


def test(model, test_data, loss_function):
    # test over complete test data

    test_accuracy_aggregator = []
    test_loss_aggregator = []

    for (input, target) in test_data:
        prediction = model(input)
        sample_test_loss = loss_function(target, prediction)
        sample_test_accuracy = np.argmax(target, axis=1) == np.argmax(prediction, axis=1)
        sample_test_accuracy = np.mean(sample_test_accuracy)
        test_loss_aggregator.append(sample_test_loss.numpy())
        test_accuracy_aggregator.append(np.mean(sample_test_accuracy))

    test_loss = tf.reduce_mean(test_loss_aggregator)
    test_accuracy = tf.reduce_mean(test_accuracy_aggregator)

    return test_loss, test_accuracy



train_dataset = train_ds.apply(prepare_FashionMNIST_data)
test_dataset = test_ds.apply(prepare_FashionMNIST_data)


tf.keras.backend.clear_session()


### Hyperparameters
num_epochs = 10
learning_rate = 0.001

# Initialize the model.
model = FashionMNISTConvModel()
# Initialize the loss: categorical cross entropy. Check out 'tf.keras.losses'.
cross_entropy_loss = tf.keras.losses.CategoricalCrossentropy()
# Initialize the optimizer: SGD with default parameters. Check out 'tf.keras.optimizers'
optimizer = tf.keras.optimizers.Adam(learning_rate)

# Initialize lists for later visualization.
train_losses = []

test_losses = []
test_accuracies = []

# testing once before we begin
test_loss, test_accuracy = test(model, test_dataset, cross_entropy_loss)
test_losses.append(test_loss)
test_accuracies.append(test_accuracy)

# check how model performs on train data once before we begin
train_loss, _ = test(model, train_dataset, cross_entropy_loss)
train_losses.append(train_loss)

# We train for num_epochs epochs.
for epoch in range(num_epochs):
    print(f'Epoch: {str(epoch)} starting with accuracy {test_accuracies[-1]}')

    # training (and checking in with training)
    epoch_loss_agg = []
    for input, target in train_dataset:
        train_loss = train_step(model, input, target, cross_entropy_loss, optimizer)
        epoch_loss_agg.append(train_loss)

    # track training loss
    train_losses.append(tf.reduce_mean(epoch_loss_agg))

    # testing, so we can track accuracy and test loss
    test_loss, test_accuracy = test(model, test_dataset, cross_entropy_loss)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)

# Visualize accuracy and loss for training and test data.
plt.figure()
line1, = plt.plot(train_losses)
line2, = plt.plot(test_losses)
line3, = plt.plot(test_accuracies)
plt.xlabel("Training steps")
plt.ylabel("Loss/Accuracy")
plt.legend((line1, line2, line3), ("training", "test", "test accuracy"))
plt.show()
