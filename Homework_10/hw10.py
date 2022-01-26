import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
import tensorflow as tf
import tensorflow_text as tf_txt
from collections import Counter
from pathlib import Path
import re


def handle_special_and_lower(text):
    return re.sub(r'[^a-z\s]', '', text.replace('\n', '').lower())


def tokenize_text(text):
    sp = tf_txt.WhitespaceTokenizer()
    return sp.split(text).numpy()


def build_token_to_index(vocab):
    return {t: i for t, i in zip(vocab, range(len(vocab)))}


def preprocess_text(text, bow_size=10000, window_size=4):
    text = handle_special_and_lower(text)
    tokens = tokenize_text(text)
    common_tokens = dict(Counter(tokens).most_common(bow_size)).keys()
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
    return (value, target), t2i


def prepare_data(ds):
    return ds.cache().shuffle(10000).batch(32).prefetch(32)


class WordEmbedding(Dense):

    def __init__(self, embedding_size):
        super(WordEmbedding, self).__init__(embedding_size, use_bias=False)

    def word_to_vec(self, one_hot_word_index):
        return self.weights[0][one_hot_word_index]


def scip_gram_model(embedding_size, bow_size):
    inp = Input(shape=(bow_size))
    emb = WordEmbedding(embedding_size)
    x = emb(inp)
    out = Dense(bow_size, activation=tf.nn.softmax)(x)
    return tf.keras.Model(inp, out), emb


bow_size = 10000
embedding_size = 128

data = Path('bible.txt').read_text()
(data, t2i) = preprocess_text(data, bow_size)

train_dataset = tf.data.Dataset.from_tensor_slices(data).map(
    lambda v, t: (tf.one_hot(v, depth=bow_size), tf.one_hot(t, depth=bow_size))).apply(prepare_data)

tf.keras.backend.clear_session()

model, emb = scip_gram_model(embedding_size, bow_size)

model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])
model.fit(train_dataset, epochs=10)

emb.word_to_vec(t2i[b'king'])
