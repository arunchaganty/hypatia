#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions
"""

import os
import json
import itertools
import logging
from collections import namedtuple
import numpy as np
from numpy import array

def _json_object_hook(d): return namedtuple('X', d.keys())(*d.values())
def json2obj(data): return json.loads(data, object_hook=_json_object_hook)

def process_snli_data(datafile):
    """
    Read the SNLI data and return output as ((sentence1, sentence2), label)
    """
    example = namedtuple('Example', ["sentence1", "sentence2", "label"])
    for line in datafile:
        obj = json.loads(line)
        tokens1 = obj["sentence1_binary_parse"].replace(r'(', '').replace(r')', '').split()
        tokens2 = obj["sentence2_binary_parse"].replace(r'(', '').replace(r')', '').split()
        label = obj["gold_label"]
        if label == "-": continue # Skip
        yield example(tokens1, tokens2, label)

def pad_zeros(arr, length):
    return arr[:length] + [0] * max(0, length - len(arr))

def load_data(datafile, max_length=40):
    """
    Returns a vectorized representation of the input data.
    Each sentence is converted into a sequence of token indices.
    Each output vector is translated as a one hot vector
    """

    label_map = {
        "neutral" : 0,
        "contradiction": 1,
        "entailment": 2,
        }

    X1, X2, Y = [], [], []

    logging.info("Reading data")
    for obj in process_snli_data(datafile):
        x1 = pad_zeros(WordEmbeddings.project_sentence(obj.sentence1), max_length)
        x2 = pad_zeros(WordEmbeddings.project_sentence(obj.sentence2), max_length)
        y = [0, 0, 0]
        y[label_map[obj.label]] = 1
        X1.append(x1)
        X2.append(x2)
        Y.append(y)
    logging.info("Done.")

    return array(X1), array(X2), array(Y)

class Scorer(object):
    """
    This object keeps running track of scores while training.
    """
    def __init__(self, model):
        self.metrics = model.metrics_names
        self.score = [0. for _ in self.metrics]
        self.n = 0

    def update(self, score, n_items):
        """
        Update metrics
        """
        self.n += n_items
        for i, val in enumerate(score):
            self.score[i] += (n_items * score[i] - val)/self.n

    def __str__(self):
        return "\t".join(str(name) + " " + str(score) for name, score in zip(self.metrics, self.score))

    def keys(self):
        """
        Return metric types
        """
        return self.metrics

    def values(self):
        """
        Return values of the matrics as a string.
        """
        return self.score

def grouper(n, iterable):
    """
    grouper(3, 'ABCDEFG') --> 'ABC', 'DEF', 'G'
    """
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk

class WordEmbeddings(object):
    """
    A wrapper around a word vector model.
    """
    class __WordEmbeddings(dict): 
        """
        Singleton class for word embeddings.
        """
        def __init__(self, index_map, weights, dim, preserve_case=False, unknown='unk'):
            dict.__init__(self, index_map)
            self.weights = weights
            self.dim = dim
            self.preserve_case = preserve_case
            self.unknown = unknown

        def __getitem__(self, key):
            if not self.preserve_case:
                key = str.lower(key)
            try:
                return dict.__getitem__(self, key)
            except KeyError:
                return dict.__getitem__(self, self.unknown)

        def __setitem__(self, key, val):
            if not self.preserve_case:
                key = str.lower(key)
            return dict.__setitem__(self, key, val)

    instance = None
    def __init__(self):
        if WordEmbeddings.instance is None:
            raise AttributeError("Word embeddings must first be initalized using from_file")

    def __getattr__(self, name):
        """Dispatch all attribute calls to singleton instance"""
        return getattr(self.instance, name)

    def __len__(self):
        return len(self.instance)

    @classmethod
    def from_file(cls, f, preserve_case=False, unknown="unk", mmap_fname=".mmap", index_fname='.index'):
        """
        Construct a word vector map from a file
        """
        logging.info("Reading word vectors")
        if os.path.exists(index_fname) and os.path.exists(mmap_fname):
            with open(index_fname) as f:
                obj = json.load(f)
            logging.info("Using cached version on disk.")
            weights = np.memmap(mmap_fname, mode='r', shape=(len(obj["indices"]), obj["dim"]), dtype=np.float32)

            cls.instance = WordEmbeddings.__WordEmbeddings(obj["indices"], weights, obj["dim"], obj["preserve_case"], obj["unknown"])
        else:
            mapping = {}
            dim = None
            for line in f:
                parts = line.split()
                tok = parts[0]
                vec = array([float(x) for x in parts[1:]])
                if dim is None:
                    dim = len(vec)
                assert dim == len(vec), "Incorrectly sized vector"
                mapping[tok] = vec
            assert unknown in mapping, "Unknown token not defined in word vectors"

            # Create an index map and compress dictionary into a matrix.
            indices = {}
            weights = np.memmap(mmap_fname, mode='w+', shape=(len(mapping), dim), dtype=np.float32)
            for i, (key, vec) in enumerate(mapping.items()):
                indices[key] = i
                weights[i,:] = vec

            with open(index_fname, 'w') as f:
                json.dump({
                    "indices": indices,
                    "dim":dim,
                    "preserve_case":preserve_case,
                    "unknown":unknown,
                    }, f)

            cls.instance = WordEmbeddings.__WordEmbeddings(indices, weights, dim, preserve_case, unknown)
            logging.info("Done. Loaded %d vectors.", len(cls.instance.weights))

    @classmethod
    def from_filename(cls, fname, preserve_case=False, unknown="unk"):
        """
        Construct a word vector map from a file
        """
        with open(fname) as f:
            mmap_fname = fname+'.mmap'
            WordEmbeddings.from_file(f, preserve_case, unknown, mmap_fname)

    @classmethod
    def project_sentence(cls, toks, max_length=40):
        """
        Return a list of indices into the word vectors
        """
        return [cls.instance[t] for t in toks[:max_length]]

    @classmethod
    def embed_sentence(cls, indices, max_length=40):
        """
        Return the list of tokens embedded as a matrix.
        """
        return cls.instance.weights[indices[:max_length],:]

    @classmethod
    def embed_sentences(cls, sentences, max_length=40):
        """
        Return the list of tokens embedded as a matrix.
        """
        return array([[cls.embed_sentence(toks, max_length)] for toks in sentences])


