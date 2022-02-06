import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAvgPool1D
from SentencePieceEmbedding import SentencePieceEmbedding
from Transformer import Transformer


class TextGeneratorModel(tf.keras.Model):

    def __init__(self, tokenizer, sequence_len, vocab_sice, embedding_sice=256):
        super(TextGeneratorModel, self).__init__()
        self.sequence_len = sequence_len
        self.tokenizer = tokenizer
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

    def generate_text(self, start_text):
        tokens = self.tokenizer.tokenize(start_text)
        tokens_len = tokens.shape()[-1]
        if tokens_len >= self.sequence_len:
            raise Exception("Sequence to long")
        for i in range(tokens_len, self.sequence_len + 1):
            inp = self.__pad_batch(tokens)
            prob_tokens = self.call(inp)
            next_token = self.__to_token(prob_tokens)
            tokens = tf.concat([tokens, [next_token]], axis=-1)
        result = self.tokenizer.detokenize(tokens)
        return result

    def __pad_batch(self, tokens):
        # self.sequence_len
        result = tokens  # TODO
        return result

    def __to_token(self, pred):
        return pred
