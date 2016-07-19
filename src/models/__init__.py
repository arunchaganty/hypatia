#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Models
"""
from functools import lru_cache
from keras.models import model_from_json, Model

class EntailmentModel(Model):
    """An entailment model"""

    output_shape = 3 # [neutral, entailment, contradiction]

    def __init__(self, x1, x2, y, **kwargs):
        """
        @x1: Input - sentence 1
        @x2: Input - sentence 2
        @y:  Output - multiclass output
        """
        super(EntailmentModel, self).__init__(input=[x1, x2], output=[y])

#    @lru_cache
#    def build(self):
#        """Build the model"""
#        pass
#
#    # predict
#    @lru_cache
#    def predict(self, examples):
#        """Predict output."""
#        pass
#
#    def train(self, examples):
#        """
#        Train an entailment model.
#        """
#        pass
#
#    @classmethod
#    def load(cls, fname_prefix):
#        """
#        Load model and weights from a file.
#        """
#        model_fname = fname_prefix = ".model"
#        weights_fname = fname_prefix = ".weights"
#
#        with open(model_fname, "r") as f:
#            json = f.read()
#        model = model_from_json(json)
#        model.load_weights(weights_fname)
#
#        return model
#
#    def save(self, fname_prefix):
#        """
#        Save the model and weights in a file.
#        """
#        assert self._model is not None, "Model must first be built."
#
#        model_fname = fname_prefix = ".model"
#        weights_fname = fname_prefix = ".weights"
#
#        with open(model_fname, "r") as f:
#            f.write(self._model.to_json())
#        self._model.save_weights(weights_fname)
#
