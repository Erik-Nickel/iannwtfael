import tensorflow as tf
from tensorflow.keras.layers import Dense, Layer, Input
from collections import Counter
import tensorflow as tf
import tensorflow_text as tf_txt
from pathlib import Path
import re


def handle_special_and_lower(text):
    return re.sub(r'[^a-z\s]', '', text.replace('\n', '').lower())


def tokenize_text(text):
    sp = tf_txt.WhitespaceTokenizer()
    return sp.split(text).numpy()


def build_index_to_token(vocab):
    return {i: t for t, i in zip(vocab, range(len(vocab)))}


def build_token_to_index(vocab):
    return {t: i for t, i in zip(vocab, range(len(vocab)))}


def preprocess_text(text, bow_size=10000, window_size=4):
    text = handle_special_and_lower(text)
    tokens = tokenize_text(text)
    common_tokens = dict(Counter(tokens).most_common(bow_size)).keys()
    i2t = build_index_to_token(common_tokens)
    t2i = build_token_to_index(common_tokens)
    value = []
    target = []
    for to, i in zip(tokens, range(len(tokens))):
        if to in t2i:
            for j in range(int(window_size / 2)):
                if i + j + 1 <= len(tokens) - 1:
                    ta = tokens[i + j + 1]
                    if ta in t2i:
                        value.append(t2i[to])
                        target.append(t2i[ta])
                if i - j - 1 >= 0:
                    ta = tokens[i - j - 1]
                    if ta in t2i:
                        value.append(t2i[to])
                        target.append(t2i[ta])
    return (value, target), i2t, t2i


def prepare_data(ds):
    return ds.cache().shuffle(10000).batch(32).prefetch(32)


class WordEmbedding(Dense):

    def __init__(self, embedding_size):
        super(WordEmbedding, self).__init__(embedding_size, use_bias=False)

    def word_to_vec(self, one_hot_word):
        pass  # TODO


def scip_gram_model(embedding_size, bow_size):
    inp = Input(shape=(bow_size))
    x = WordEmbedding(embedding_size)(inp)
    out = Dense(bow_size, activation=tf.nn.softmax)(x)
    return tf.keras.Model(inp, out)


bow_size = 10000

data = Path('bible.txt').read_text()
(data, i2t, t2i) = preprocess_text(data, bow_size)

train_dataset = tf.data.Dataset.from_tensor_slices(data).map(
    lambda v, t: (tf.one_hot(v, depth=len(t2i)), tf.one_hot(t, depth=len(t2i)))).apply(prepare_data)

print(train_dataset.take(1))

tf.keras.backend.clear_session()

model = scip_gram_model(100, bow_size)
model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_dataset, epochs=10)
