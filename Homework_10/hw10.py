import tensorflow as tf
from tensorflow.keras.layers import Dense, Layer, Input
from collections import Counter
import tensorflow as tf
import tensorflow_text as tf_txt
from pathlib import Path
import numpy as np
import re


def handle_special_and_lower(text):
    return re.sub(r'[^a-z\s]', '', text.replace('\n', '').lower())


def tokenize_text(text):
    sp = tf_txt.WhitespaceTokenizer()
    return sp.split(text).numpy()


def preprocess_text(text, bow_size=10000):
    text = handle_special_and_lower(text)
    tokens = tokenize_text(text)
    common_tokens = dict(Counter(tokens).most_common(bow_size)).keys()
    # TODO: Build Target label pairs
    return tokens, common_tokens


def prepare_data(ds):
    return ds.cache().shuffle(10000).batch(32).prefetch(32)


class WordEmbedding(Dense):

    def __init__(self, embedding_size):
        super(WordEmbedding, self).__init__(embedding_size, use_bias=False)


def scip_gram_model(embedding_size, bow_size):
    inp = Input(shape=(bow_size))
    x = WordEmbedding(embedding_size)(inp)
    out = Dense(bow_size, activation=tf.nn.softmax)(x)
    return tf.keras.Model(inp, out)


data = Path('bible.txt').read_text()
(data, bow_tokens) = preprocess_text(data)
train_dataset = tf.data.Dataset.from_tensor_slices(data).apply(prepare_data)

bow_size = None  # TODO

tf.keras.backend.clear_session()

model = scip_gram_model(100, bow_size)
model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_dataset, epochs=10)
