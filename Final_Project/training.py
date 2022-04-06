from FoodRecommenderModel import FoodRecommenderModel
from FoodRatingDataSet import FoodRatingDataset
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint
from datetime import datetime
import tensorflow as tf
import os.path

NUM_RECIPES = 178265  # 163690 #161880
SEQ_LEN = 9
__CHECKPOINT_DIR = "checkpoints/checkpoint"


def model_summary():
    mod = create_model(SEQ_LEN)
    ids = tf.keras.Input(shape=[SEQ_LEN])
    mod(ids)
    mod.summary()


def create_model(seq_len):
    return FoodRecommenderModel(recipe_count=NUM_RECIPES, seq_len=seq_len)


def train(data, model):
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])

    board_file_name = datetime.now().strftime("%d%m%Y_%H%M%S")

    if os.path.exists(__CHECKPOINT_DIR):
        print("load model:")
        board_file_name = f"{board_file_name}_loaded"
        load_wights(model)
    else:
        print("new model:")

    tensorboard = TensorBoard(log_dir=f"logs/recommend_{board_file_name}")
    checkpoints = ModelCheckpoint(filepath=__CHECKPOINT_DIR, save_weights_only=True)

    (train_ds, val_ds) = data
    model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=[tensorboard, checkpoints])


def load_wights(model):
    model.load_weights(__CHECKPOINT_DIR)


def run():
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~TRAINING~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    dataset = FoodRatingDataset(seq_len=SEQ_LEN)
    data = dataset.data()
    print("DATA: ", data)
    model_summary()
    model = create_model(SEQ_LEN)
    train(data, model)


if __name__ == "__main__":
    run()
