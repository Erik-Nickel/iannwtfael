
import numpy as np
import pandas as pd
#import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense


df = pd.read_csv('C:/Users/enick/Documents/1-University/master/ws21/IANNwTF/datasets/winequality-red.csv',sep = ";")
print(df.head())
print(df.mean())
print(df.max() -df.min())


df.quality = np.where(df.quality > df.quality.mean(),np.int64(1),np.int64(0))
df = df.sample(frac=1).reset_index(drop=True)
target = df.pop('quality')



#sns.set()
#sns.pairplot(df)


print(target)



target_train = target.iloc[0:int(len(target)* 0.7)]
target_validation = target.iloc[int(len(target)* 0.7):int(len(target)* 0.85)]
target_test =target.iloc[int(len(target)* 0.85):len(target)]
df_train = df.iloc[0:int(len(df)* 0.7)]
df_validation = df.iloc[int(len(df)* 0.7):int(len(df)* 0.85)]
df_test =df.iloc[int(len(df)* 0.85):len(df)]


train_dataset = tf.data.Dataset.from_tensor_slices((df_train.values, target_train.values))
validation_dataset = tf.data.Dataset.from_tensor_slices((df_validation.values, target_validation.values))
test_dataset = tf.data.Dataset.from_tensor_slices((df_test.values, target_test.values))



def prepare_wine(wine):
    wine = wine.map(lambda point, label: (point, tf.one_hot(label, 2)))
    wine = wine.shuffle(10000000)
    wine = wine.batch(32)
    return wine



train_ds = train_dataset.apply(prepare_wine)
test_ds = test_dataset.apply(prepare_wine)


for elem,label in train_dataset.take(3):
    tf.print(elem,label)


class WineModel(tf.keras.Model):

    def __init__(self):
        super(WineModel, self).__init__()
        self.dense1 = Dense(10, activation=tf.nn.sigmoid)
        self.dense2 = Dense(256, activation=tf.nn.sigmoid)
        self.out = Dense(2, activation=tf.nn.sigmoid)

    @tf.function
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.out(x)
        return x



def train_step(model, input, target, loss_function, optimizer):
    with tf.GradientTape() as tape:
        prediction = model(input)
        loss = loss_function(target, prediction)
        gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


def test(model, test_data, loss_function):
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


tf.keras.backend.clear_session()

#train_ds = train_ds.take()
#test_ds = test_ds.take()

num_epochs = 10
learning_rate = 0.1

model = WineModel()
cross_entropy_loss = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

train_losses = []
test_losses = []
test_accuracies = []

test_loss, test_accuracy = test(model, test_dataset, cross_entropy_loss)
test_losses.append(test_loss)
test_accuracies.append(test_accuracy)

train_loss, _ = test(model, train_dataset, cross_entropy_loss)
train_losses.append(train_loss)

for epoch in range(num_epochs):
    print(f'Epoch: {str(epoch)} starting with accuracy {test_accuracies[-1]}')
    epoch_loss_agg = []
    for input, target in train_dataset:
        train_loss = train_step(model, input, target, cross_entropy_loss, optimizer)
        epoch_loss_agg.append(train_loss)
    train_losses.append(tf.reduce_mean(epoch_loss_agg))

    test_loss, test_accuracy = test(model, test_dataset, cross_entropy_loss)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)



plt.plot(train_losses, label="training")
plt.plot(test_losses, label="test")
plt.plot(test_accuracies, label="test accuracy")
plt.xlabel("Training steps")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()


