#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A basic entailment model
"""
from . import EntailmentModel
from keras import backend as K
from keras.layers import Dense, Dropout, Activation, Reshape, Flatten, Merge, Input, AveragePooling1D, merge, Embedding
from util import WordEmbeddings

# TODO: create a sentence embedding entailment model.
class BasicModel(EntailmentModel):
    """
    The basic model encodes both sentences using average pooling and
    trains a single layer model on the sentence encodings.
    """

    def __init__(self, input_length, output_type="sigmoid", **kwargs):
        emb = Embedding(len(WordEmbeddings()), WordEmbeddings().dim, weights=[WordEmbeddings().weights], trainable=False)

        # The two sentece inputs
        x1, x2 = Input(shape=(input_length, WordEmbeddings().dim,), dtype="float32"), Input(shape=(input_length, WordEmbeddings().dim,), dtype="float32")

        # Average pool
        h1, h2 = AveragePooling1D(pool_length=input_length-1)(x1), AveragePooling1D(pool_length=input_length-1)(x2)

        # Softmax on top of these.
        z = merge([h1,h2], mode="concat")
        z = Flatten()(z)
        z = Dropout(0.5)(z)
        y = Dense(self.output_shape, activation=output_type)(z)

        super(BasicModel, self).__init__(x1, x2, y, **kwargs)

