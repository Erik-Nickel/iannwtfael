import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, MultiHeadAttention, LayerNormalization, Dropout, Add, GlobalAvgPool1D
from SentencePieceEmbedding import SentencePieceEmbedding
from Transformer import Transformer


class TextGeneratorModel(tf.keras.Model):

    def __init__(self, tokenizer, sequence_len, vocab_sice=7000, embedding_sice=256):
        super(TextGeneratorModel, self).__init__()
        self.emb = SentencePieceEmbedding(vocab_sice, embedding_sice, sequence_len)
        self.trans = Transformer(embedding_sice)
        self.pooling = GlobalAvgPool1D()
        self.dense = Dense(vocab_sice)

    @tf.function
    def call(self, inputs, is_training=False):
        x = self.emb(inputs)
        x = self.trans(x)
        x = self.pooling(x)
        x = self.dense(x)
        return x
