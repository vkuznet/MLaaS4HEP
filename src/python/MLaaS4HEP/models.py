#!/usr/bin/env python
#-*- coding: utf-8 -*-
#pylint: disable=R0913,R0914
"""
File       : models.py
Author     : Valentin Kuznetsov <vkuznet AT gmail dot com>
Models module defines wrappers to train MLaaS4HEP workflows.

User should provide a model implementation (so far in either in Keras or PyTorch),
see ex_keras.py and ex_pytorch.py, respectively for examples.

The Trainer class defines a thin wrapper around user model and provides uniformat
APIs to fit the model and yield predictions.
"""
from __future__ import print_function, division, absolute_import

# system modules

# numpy modules
import numpy as np

# keras modules
from keras.utils import to_categorical

# pytorch modules
try:
    import torch
except ImportError:
    torch = None

# MLaaS4HEP modules
from MLaaS4HEP.generator import RootDataGenerator, MetaDataGenerator, file_type
from MLaaS4HEP.utils import load_code

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

    def fit(self, x_train, y_train, **kwds):
        """
        Fit API of the trainer.

        :param data: the ROOT IO data in form of numpy array of data and mask vectors.
        :param y_train: the true values vector for input data.
        :param kwds: defines input set of parameters for end-user model.
        """
        if self.verbose:
            print("Perform fit on {} data with {}"\
                    .format(np.shape(x_train), kwds))
        if self.cls_model.find('keras') != -1:
            self.model.fit(x_train, y_train, verbose=self.verbose, **kwds)
        elif self.cls_model.find('torch') != -1:
            self.model(x_train).data.numpy()
        else:
            raise NotImplementedError

    def predict(self):
        "Predict API of the trainer"
        raise NotImplementedError

    def save(self, fout):
        "Save our model to given file"
        if self.cls_model.find('keras') != -1:
            self.model.save(fout)
        elif self.cls_model.find('torch') != -1 and torch:
            torch.save(self.model, fout)
        else:
            raise NotImplementedError

def train_model(model, files, labels, preproc=None, params=None, specs=None, fout=None, dtype=None):
    """
    Train given model on set of files, params, specs

    :param model: the model class
    :param files: the list of files to use for training
    :param labels: the list of label files to use for training or label name to use in data
    :param preproc: file name which contains preprocessing function
    :param params: list of parameters to use for training (via input json file)
    :param specs: file specs
    :param fout: output file name to save the trained model
    """
    if not params:
        params = {}
    if not specs:
        specs = {}
    model = load_code(model, 'model')
    if preproc:
        preproc = load_code(preproc, 'preprocessing')
    if file_type(files) == 'root':
        gen = RootDataGenerator(files, labels, params, specs)
    else:
        gen = MetaDataGenerator(files, labels, params, preproc, dtype)
    epochs = params.get('epochs', 10)
    batch_size = params.get('batch_size', 50)
    shuffle = params.get('shuffle', True)
    split = params.get('split', 0.3)
    trainer = False
    kwds = {'epochs':epochs, 'batch_size': batch_size,
            'shuffle': shuffle, 'validation_split': split}
    for data in gen:
        if np.shape(data[0])[0] == 0:
            print("received empty x_train chunk")
            break
        if len(data) == 2:
            x_train = data[0]
            y_train = data[1]
        elif len(data) == 3: # ROOT data with mask array
            x_train = data[0]
            x_mask = data[1]
            x_train[np.isnan(x_train)] = 0 # convert all nan's to zero
            y_train = data[2]
            print("x_mask chunk of {} shape".format(np.shape(x_mask)))
        print("x_train chunk of {} shape".format(np.shape(x_train)))
        print("y_train chunk of {} shape".format(np.shape(y_train)))
        if not trainer:
            idim = np.shape(x_train)[-1] # read number of attributes we have
            model = model(idim)
            print("model", model, "loss function", model.loss)
            trainer = Trainer(model, verbose=params.get('verbose', 0))
        # convert y_train to categorical array
        if model.loss == 'categorical_crossentropy':
            y_train = to_categorical(y_train)
        trainer.fit(x_train, y_train, **kwds)

    if fout and hasattr(trainer, 'save'):
        trainer.save(fout)
