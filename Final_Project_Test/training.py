from FoodRecommenderModelSequence import FoodRecommenderModelSequence
from FoodRatingDataSet import FoodRatingDataset
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint
from datetime import datetime
import tensorflow as tf
import os.path

NUM_RECIPES = 178265  # 163690 #161880
NUM_ING = 8023
OTHER_FEATURES = 3
SEQ_LEN = 9
BATCH_SIZE = 178265 #64





def model_summary():
    mod = create_model(SEQ_LEN)
    ids, ing, ofe = tf.keras.Input(shape=[SEQ_LEN]), tf.keras.Input(shape=[SEQ_LEN, NUM_ING]), tf.keras.Input(
        shape=[SEQ_LEN, OTHER_FEATURES])
    mod((ids, ing, ofe))
    mod.summary()


def create_model(seq_len):
    return FoodRecommenderModelSequence(recipe_count=NUM_RECIPES, seq_len=seq_len)


def train(data, model):
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])

    board_file_name = datetime.now().strftime("%d%m%Y_%H%M%S")
    checkpoints_dir = "checkpoints/checkpoint"

    if os.path.exists(checkpoints_dir):
        print("load model:")
        board_file_name = f"{board_file_name}_loaded"
        model.load_weights(checkpoints_dir)
    else:
        print("new model:")

    tensorboard = TensorBoard(log_dir=f"logs/recommend_{board_file_name}")
    checkpoints = ModelCheckpoint(filepath=checkpoints_dir, save_weights_only=True)

    # model.fit(data, epochs=10, validation_split=0.3, callbacks=[tensorboard, checkpoints])
    model.fit(data, epochs=10, callbacks=[tensorboard, checkpoints])


def run():
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~TRAINING~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    dataset = FoodRatingDataset()
    data = dataset.data()
    print("DATA: ", data)
    model_summary()
    model = create_model(SEQ_LEN)
    train(data, model)


if __name__ == "__main__":
    run()
