#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Models
"""
from functools import lru_cache
from keras.models import model_from_json, Model
from keras.layers import Input, AveragePooling1D

class StorableModel(Model):
    """A model that can be stored."""

    @classmethod
    def load(cls, fname_prefix):
        """
        Load model and weights from a file.
        """
        model_fname = fname_prefix + ".model"
        weights_fname = fname_prefix + ".weights"

        with open(model_fname, "r") as f:
            json = f.read()
        model = model_from_json(json, custom_objects={cls.__name__:cls})
        model.load_weights(weights_fname)

        return model

    def save(self, fname_prefix):
        """
        Save the model and weights in a file.
        """
        model_fname = fname_prefix + ".model"
        weights_fname = fname_prefix + ".weights"

        with open(model_fname, "w") as f:
            f.write(self.to_json())
        self.save_weights(weights_fname, overwrite=True)

    @classmethod
    def build(cls, **kwargs):
        """
        Build the model.
        @returns - Model.
        """
        raise NotImplementedError()

class EntailmentModel(StorableModel):
    """An entailment model"""
    output_shape = 3 # [neutral, entailment, contradiction]

    def __init__(self, **kwargs):
        super(EntailmentModel, self).__init__(**kwargs)

class SentenceModel(StorableModel):
    """An sentence embedding model"""

    def __init__(self, **kwargs):
        super(SentenceModel, self).__init__(**kwargs)

from models.BasicSentenceModel import BasicSentenceModel

class SentenceEntailmentModel(EntailmentModel):
    """An entailment model that uses a sentence model"""
    def __init__(self, **kwargs):
        super(SentenceEntailmentModel, self).__init__(**kwargs)
    sentence_model = BasicSentenceModel

    @classmethod
    def combine_sentences(cls, x1, x2, **kwargs):
        """
        Combine the sentence embeddings x1, x2 to produce an entailment.
        Returns variable (not Model!)
        """
        raise NotImplementedError()

    @classmethod
    def build(cls, **kwargs):
        input_shape = kwargs['input_shape']
        x1 = Input(shape=input_shape)
        x2 = Input(shape=input_shape)

        sentence_model = kwargs.get('sentence_model', cls.sentence_model).build(**kwargs)
        y1, y2 = sentence_model([x1]), sentence_model([x2])
        z = cls.combine_sentences(y1, y2, **kwargs)
        return SentenceEntailmentModel(input=[x1, x2], output=[z])

