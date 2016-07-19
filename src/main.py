#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import csv
import sys
from importlib import import_module
import logging

from numpy import array
from tqdm import tqdm
from util import load_data, grouper, Scorer, WordEmbeddings

logging.basicConfig(level=logging.DEBUG)

def init_resources(args):
    """
    Initialize any global resources (e.g. word embedding models)
    """
    WordEmbeddings.from_file(
        args.wvecs,
        mmap_fname=args.wvecs.name + ".mmap",
        index_fname=args.wvecs.name + ".index",
        )

def get_model(model):
    """import model"""
    return getattr(import_module('models.{0}'.format(model)), model)

def do_train(args):
    """
    Train a model.
    """
    X1_train, X2_train, y_train = load_data(args.train_data, args.input_length)
    X1_dev, X2_dev, y_dev = load_data(args.dev_data, args.input_length)
    model = get_model(args.model)(args.input_length)

    model.compile(
        optimizer='rmsprop',
        loss='mse',
        metrics=['accuracy'])

    for epoch in range(args.n_epochs):
        logging.info("Training model, epoch %d", epoch)

        scorer = Scorer(model)
        for xy in tqdm(grouper(args.batch_size, zip(X1_train, X2_train, y_train)), total=len(y_train)/args.batch_size):
            X1_batch, X2_batch, y_batch = zip(*xy)
            X1_batch, X2_batch, y_batch = array(X1_batch), array(X2_batch), array(y_batch)

            score = model.train_on_batch([X1_batch, X2_batch], y_batch)
            scorer.update(score, len(y_batch))
        logging.info("train error: %s", scorer)

        scorer = Scorer(model)
        for xy in tqdm(grouper(args.batch_size, zip(X1_dev, X2_dev, y_dev))):
            X1_batch, X2_batch, y_batch = zip(*xy)
            X1_batch, X2_batch, y_batch = array(X1_batch), array(X2_batch), array(y_batch)

            score = model.test_on_batch([X1_batch, X2_batch], y_batch)
            scorer.update(score, len(y_batch))
        logging.info("val error: %s", scorer)
        model.save(args.output)

def do_evaluate(args):
    """
    Evaluate an existing model.
    """
    raise NotImplementedError()

def do_console(args):
    """
    Interact with an existing model using a console.
    """
    raise NotImplementedError()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train and test a natural langauge inference model.')
    parser.add_argument('--wvecs', type=argparse.FileType('r'), default="deps/glove.6B/glove.6B.50d.txt", help="Path to word vectors.")
#    parser.add_argument('--log', type=argparse.FileType('w'), default="{rundir}/log", help="Where to log output.")
    parser.add_argument('--input_length', type=int, default=150, help="Maximum number of tokens.")

    subparsers = parser.add_subparsers()

    command_parser = subparsers.add_parser('train', help='Train a model' )
    command_parser.set_defaults(func=do_train)
    command_parser.add_argument('--train_data', type=argparse.FileType('r'), default="data/snli_1.0/snli_1.0/snli_1.0_train.jsonl", help="Path to SNLI training data.")
    command_parser.add_argument('--dev_data', type=argparse.FileType('r'), default="data/snli_1.0/snli_1.0/snli_1.0_dev.jsonl", help="Path to SNLI dev data.")
    command_parser.add_argument('--model', choices=["BasicModel",], default="BasicModel", help="Type of model to use.")
    command_parser.add_argument('--n_epochs', type=int, default=10, help="Number of training passes.")
    command_parser.add_argument('--batch_size', type=int, default=64, help="Size of minibatch")
#    command_parser.add_argument('--eval_output', type=argparse.FileType('w'), default="{rundir}/eval", help="Evaluation output.")
#    command_parser.add_argument('--output', type=argparse.FileType('w'), default="{rundir}/output", help="Type of model to use.")
    command_parser.add_argument('--output', default="output", help="Type of model to use.")
    command_parser.add_argument('params', nargs="*", help="Additional parameters for the model")

    command_parser = subparsers.add_parser('evaluate', help='Evaluate a model on a dataset.' )
    command_parser.set_defaults(func=do_evaluate)
    command_parser.add_argument('--model', type=str, default="{rundir}/model", help="Model to use for evaluation.")
    command_parser.add_argument('--eval_data', type=argparse.FileType('r'), default="data/snli_1.0/", help="Data to evaluate the model on.")
#    command_parser.add_argument('--output', type=argparse.FileType('r'), default="{rundir}/eval", help="Evaluation output.")

    command_parser = subparsers.add_parser('console', help='Run a console to interact with the model.' )
    command_parser.set_defaults(func=do_console)
#    command_parser.add_argument('--model', type=str, default="{rundir}/model", help="Model to use for evaluation.")

    ARGS = parser.parse_args()
    init_resources(ARGS)
    ARGS.func(ARGS)
