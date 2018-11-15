#!/usr/bin/env python
#-*- coding: utf-8 -*-
#pylint: disable=
"""
File       : models.py
Author     : Valentin Kuznetsov <vkuznet AT gmail dot com>
Description: this module defines handles based ML models for MLaaS4HEP workflow.

User should provide a model implementation (so far in either in Keras or PyTorch),
see ex_keras.py and ex_pytorch.py, respectively for examples.

The Trainer class defines a thin wrapper around user model and provides uniformat
APIs to fit the model and yield predictions.
"""

# system modules
import os
import sys

# numpy modules
import numpy as np

# MLaaS4HEP modules
from reader import xfile
from generator import DataGenerator

class Trainer(object):
    """
    Trainer class defines a think wrapper around given user model.
    It defines basic `fit` and `predict` APIs.
    """
    def __init__(self, model, verbose=0):
        self.model = model
        self.verbose = verbose
        self.cls_model = '{}'.format(type(self.model)).lower()
        if self.verbose:
            try:
                print(self.model.summary())
            except AttributeError:
                print(self.model)

    def fit(self, data, y_train, **kwds):
        """
        Fit API of the trainer.

        :param data: the ROOT IO data in form of numpy array of data and mask vectors.
        :param y_train: the true values vector for input data.
        :param kwds: defines input set of parameters for end-user model.
        """
        xdf, mask = data[0], data[1]
        # cast values in data vector according to the mask
        xdf[np.isnan(mask)] = 0
        if self.verbose:
            print("Perform fit on {} data with {}"\
                    .format(np.shape(xdf), kwds))
        if self.cls_model.find('keras') != -1:
            self.model.fit(xdf, y_train, verbose=self.verbose, **kwds)
        elif self.cls_model.find('torch') != -1:
            yhat = self.model(xdf).data.numpy()
        else:
            raise NotImplemented

    def predict(self):
        "Predict API of the trainer"
        raise NotImplemented

def load_model(mfile):
    """
    Load model from given pyton module (file)

    :param mfile: the python module name which implements model function
    """
    mname = mfile.split('.py')[0].replace('/', '.')
    try:
        mod = __import__(mname, fromlist=['model'])
        print("loaded {} {} {}".format(mfile, mod.model, mod.model.__doc__))
        return mod.model
    except ImportError:
        traceback.print_exc()
        msg = "Please provide python module which implements model function.\n"
        msg += "The input file name should be visible through PYTHONPATH"
        print(msg)
        raise

def train_model(model, files, params=None, specs=None, fout=None):
    """
    Train given model on set of files, params, specs

    :param model: the model class
    :param files: the list of files to use for training
    :param params: list of parameters to use for training (via input json file)
    :param specs: file specs
    :param fout: output file name to save the trained model
    """
    if not params:
        params = {}
    if not specs:
        specs = {}
    model = load_model(model)
    xfiles = [xfile(f) for f in files]
    gen = DataGenerator(xfiles, params, specs)
    epochs = specs.get('epochs', 10)
    batch_size = specs.get('batch_size', 50)
    shuffle = specs.get('shuffle', True)
    split = specs.get('split', 0.3)
    trainer = False
    for data in gen:
        x_train = data[0]
        x_mask = data[1]
        print("x_train chunk of {} shape".format(np.shape(x_train)))
        print("x_mask chunk of {} shape".format(np.shape(x_mask)))
        if np.shape(x_train)[0] == 0:
            print("received empty x_train chunk")
            break
        if not trainer:
            idim = np.shape(x_train)[-1] # read number of attributes we have
            trainer = Trainer(model(idim), verbose=params.get('verbose', 0))

        # TODO: the y_train should be given us externally, so far we create it as random values
        # create dummy vector for y's for our x_train
        from keras.utils import to_categorical
        y_train = np.random.randint(2, size=np.shape(x_train)[0])
        y_train = to_categorical(y_train) # convert labesl to categorical values
        print("y_train {} chunk of {} shape".format(y_train, np.shape(y_train)))
        kwds = {'epochs':epochs, 'batch_size': batch_size, 'shuffle': shuffle, 'validation_split': split}

        trainer.fit(data, y_train, **kwds)
    if fout:
        trainer.save(fout)
