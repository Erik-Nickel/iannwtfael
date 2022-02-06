import tensorflow as tf
import tensorflow_text as tf_txt
from pathlib import Path
import sentencepiece as sp
import datetime
import tqdm
import io
import re
import os
from TextGeneratorModel import TextGeneratorModel


def prepare_data(ds):
    return ds.map(set_target).shuffle(10000).batch(32).prefetch(32)


def set_target(x):
    return x[:-2], x[-1]


SEQ_LEN = 128  # 32-256
VOCAB_SIZE = 7000  # 2000-7000

if not os.path.exists("tokenizer_model.model"):
    print("[INFO] train tokenizer model")
    sp.SentencePieceTrainer.train(input='bible.txt', model_prefix='tokenizer_model', model_type="unigram",
                                  vocab_size=SEQ_LEN, pad_id=0, unk_id=3)
trained_tokenizer_model = tf.io.gfile.GFile('tokenizer_model.model', "rb").read()
tokenizer = tf_txt.SentencepieceTokenizer(model=trained_tokenizer_model, out_type=tf.int32, nbest_size=-1, alpha=1,
                                          reverse=False, add_bos=False, add_eos=False, return_nbest=False, name=None)

data = Path('bible.txt').read_text()
tokens = tokenizer.tokenize(data)
ds = tf_txt.sliding_window(data=tokens, width=SEQ_LEN + 1, axis=0)
train_dataset = tf.data.Dataset.from_tensor_slices(ds).apply(prepare_data)

tf.keras.backend.clear_session()

model = TextGeneratorModel(tokenizer, SEQ_LEN, VOCAB_SIZE)
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
print("Tran")
model.fit(train_dataset, epochs=10)
