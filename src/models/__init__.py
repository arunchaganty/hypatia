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

    def __init__(self, *args, **kwargs):
        super(EntailmentModel, self).__init__(*args, **kwargs)

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

