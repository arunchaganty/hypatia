#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import ipdb
import csv
import sys
from importlib import import_module
import logging

import numpy as np
from numpy import array
from tqdm import tqdm
from util import vectorize_data, grouper, Scorer, WordEmbeddings, process_snli_data, LABELS, LABEL_MAP, ConfusionMatrix, Example

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s]:%(levelname)s:%(message)s",
    datefmt="%m/%d/%Y %H:%M:%S")

def init_resources(args):
    """
    Initialize any global resources (e.g. word embedding models)
    """
    WordEmbeddings.from_file(
        args.wvecs,
        mmap_fname=args.wvecs.name + ".mmap",
        index_fname=args.wvecs.name + ".index",
        )

def get_model_factory(model):
    """import model"""
    return getattr(import_module('models.{0}'.format(model)), model)

def do_train(args):
    """
    Train a model.
    """
    X1_train, X2_train, y_train = vectorize_data(list(process_snli_data(args.train_data)), args.input_length)
    X1_dev, X2_dev, y_dev = vectorize_data(list(process_snli_data(args.dev_data)), args.input_length)
    logging.info("Building model")
    model = get_model_factory(args.model).build(args.input_length)

    model.compile(
        optimizer='rmsprop',
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    logging.info("Done.")

    emb = WordEmbeddings()
    logging.info("Training model")
    for epoch in range(args.n_epochs):
        logging.info("Epoch %d", epoch)

        scorer = Scorer(model)
        for xy in tqdm(grouper(args.batch_size, zip(X1_train, X2_train, y_train)), total=int(len(y_train)/args.batch_size)):
            X1_batch, X2_batch, y_batch = zip(*xy)
            X1_batch = array([emb.weights[x,:] for x in X1_batch])
            X2_batch = array([emb.weights[x,:] for x in X2_batch])
            y_batch = array(y_batch)

            score = model.train_on_batch([X1_batch, X2_batch], y_batch)
            scorer.update(score, len(y_batch))
        logging.info("train error: %s", scorer)

        scorer = Scorer(model)
        for xy in tqdm(grouper(args.batch_size, zip(X1_dev, X2_dev, y_dev)), total=int(len(y_dev)/args.batch_size)):
            X1_batch, X2_batch, y_batch = zip(*xy)
            X1_batch = array([emb.weights[x,:] for x in X1_batch])
            X2_batch = array([emb.weights[x,:] for x in X2_batch])
            y_batch = array(y_batch)

            score = model.test_on_batch([X1_batch, X2_batch], y_batch)
            scorer.update(score, len(y_batch))
        logging.info("val error: %s", scorer)
        model.save(args.output)
    logging.info("Done.")

def do_evaluate(args):
    """
    Evaluate an existing model.
    """
    logging.info("Evaluating the model.")
    model = get_model_factory(args.model).load(args.model_path)

    data = list(process_snli_data(args.eval_data))
    X1, X2, Y = vectorize_data(data, args.input_length)

    emb = WordEmbeddings()
    cm = ConfusionMatrix(LABELS)
    writer = csv.writer(args.output, delimiter="\t")
    writer.writerow(["sentence1", "sentence2", "gold_label", "guess_label", "neutral", "contradiction", "entailment"])
    for batch in tqdm(grouper(args.batch_size, zip(data, X1, X2, Y)), total=int(len(data)/args.batch_size)):
        objs, X1_batch, X2_batch, y_batch = zip(*batch)
        X1_batch = array([emb.weights[x,:] for x in X1_batch])
        X2_batch = array([emb.weights[x,:] for x in X2_batch])
        y_batch = array(y_batch)

        y_batch_ = model.predict_on_batch([X1_batch, X2_batch])

        for obj, y, y_ in zip(objs, y_batch, y_batch_):
            label = np.argmax(y)
            label_ = np.argmax(y_)
            writer.writerow([
                obj.sentence1,
                obj.sentence2,
                LABELS[label],
                LABELS[label_],
                ] + list(y_))
            cm.update(label, label_)
    cm.print_table()
    cm.summary()
    logging.info("Done.")

def do_console(args):
    """
    Interact with an existing model using a console.
    """
    model = get_model_factory(args.model).load(args.model_path)

    emb = WordEmbeddings()
    while True:
        sentence1 = input("$p> ")
        if len(sentence1) == 0: continue
        sentence2 = input("$h> ")
        if len(sentence2) == 0: continue

        tokens1, tokens2 = sentence1.split(), sentence2.split()
        ex = Example(sentence1, sentence2, tokens1, tokens2, None)
        x1, x2, _ = vectorize_data(ex, args.input_length)
        x1, x2 = array([emb.weights[x1,:]]), array([emb.weights[x2,:]])
        y_  = model.predict([x1, x2])
        label_ = np.argmax(y_)
        print("$o> %s (%s)"%(LABELS[label_], ", ".join("%s=%.2f"%(k,v) for k,v in zip(LABELS, list(y_.flatten())))))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train and test a natural langauge inference model.')
    parser.add_argument('--wvecs', type=argparse.FileType('r'), default="deps/glove.6B/glove.6B.50d.txt", help="Path to word vectors.")
#    parser.add_argument('--log', type=argparse.FileType('w'), default="{rundir}/log", help="Where to log output.")
    parser.add_argument('--input_length', type=int, default=150, help="Maximum number of tokens.")
    parser.add_argument('--model', choices=["BasicModel","BasicFrozenModel",], default="BasicModel", help="Type of model to use.")

    subparsers = parser.add_subparsers()

    command_parser = subparsers.add_parser('train', help='Train a model' )
    command_parser.set_defaults(func=do_train)
    command_parser.add_argument('--train_data', type=argparse.FileType('r'), default="data/snli_1.0/snli_1.0/snli_1.0_train.jsonl", help="Path to SNLI training data.")
    command_parser.add_argument('--dev_data', type=argparse.FileType('r'), default="data/snli_1.0/snli_1.0/snli_1.0_dev.jsonl", help="Path to SNLI dev data.")
    command_parser.add_argument('--n_epochs', type=int, default=10, help="Number of training passes.")
    command_parser.add_argument('--batch_size', type=int, default=64, help="Size of minibatch")
#    command_parser.add_argument('--eval_output', type=argparse.FileType('w'), default="{rundir}/eval", help="Evaluation output.")
#    command_parser.add_argument('--output', type=argparse.FileType('w'), default="{rundir}/output", help="Type of model to use.")
    command_parser.add_argument('--output', default="output", help="Type of model to use.")
    command_parser.add_argument('params', nargs="*", help="Additional parameters for the model")

    command_parser = subparsers.add_parser('evaluate', help='Evaluate a model on a dataset.' )
    command_parser.set_defaults(func=do_evaluate)
    command_parser.add_argument('--batch_size', type=int, default=64, help="Size of minibatch")
    command_parser.add_argument('--model_path', type=str, default="output", help="Path of serialized model")
    command_parser.add_argument('--eval_data', type=argparse.FileType('r'), default="data/snli_1.0/", help="Data to evaluate the model on.")
    command_parser.add_argument('--output', type=argparse.FileType('w'), default=sys.stdout, help="Output")

    command_parser = subparsers.add_parser('console', help='Run a console to interact with the model.' )
    command_parser.set_defaults(func=do_console)
    command_parser.add_argument('--model_path', type=str, default="output", help="Path of serialized model")
    #command_parser.add_argument('--output', type=argparse.FileType('w'), default=sys.stdout, help="Output")

    ARGS = parser.parse_args()
    init_resources(ARGS)
    ARGS.func(ARGS)
