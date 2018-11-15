#!/usr/bin/env python
#-*- coding: utf-8 -*-
#pylint: disable=
"""
File       : models.py
Author     : Valentin Kuznetsov <vkuznet AT gmail dot com>
Description: this module defines user based ML models for workflow
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
        "Fit implementation of the trainer"
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
        "Predict function of the trainer"
        pass

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

#
# test code can be used as an example how to create and run workflow
#
def test_Keras_model(input_shape):
    "Simple Keras model for testing purposes"
    from keras.models import Sequential
    from keras.layers import Dense, Activation

    model = Sequential([
        Dense(32, input_shape=input_shape),
        Activation('relu'),
        Dense(2),
        Activation('softmax'),
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def test_PyTorch_model(input_shape):
    "Simple PyTorch model for testing purposes"
    from jarray.pytorch import JaggedArrayLinear
    import torch
    model = torch.nn.Sequential(
        JaggedArrayLinear(input_shape, 5),
        torch.nn.ReLU(),
        torch.nn.Linear(5, 1),
    )
    return model

def test_Keras(files, params=None, specs=None):
    """
    Test function demonstrates workflow of setting up data generator and train the model
    over given set of files
    """
    from keras.utils import to_categorical
    if not params:
        params = {}
    if not specs:
        specs = {}
    xfiles = [xfile(f) for f in files]
    gen = DataGenerator(xfiles, params, specs)
    epochs = specs.get('epochs', 10)
    batch_size = specs.get('batch_size', 50)
    shuffle = specs.get('shuffle', True)
    split = specs.get('split', 0.3)
    trainer = False
    for data in gen:
        x_train = np.array(data[0])
        if not trainer:
            input_shape = (np.shape(x_train)[-1],) # read number of attributes we have
            trainer = Trainer(test_model(input_shape), verbose=params.get('verbose', 0))
        print("x_train {} chunk of {} shape".format(x_train, np.shape(x_train)))
        if np.shape(x_train)[0] == 0:
            print("received empty x_train chunk")
            break
        # create dummy vector for y's for our x_train
        y_train = np.random.randint(2, size=np.shape(x_train)[0])
        y_train = to_categorical(y_train) # convert labesl to categorical values
        print("y_train {} chunk of {} shape".format(y_train, np.shape(y_train)))
        kwds = {'epochs':epochs, 'batch_size': batch_size, 'shuffle': shuffle, 'validation_split': split}
        trainer.fit(data, y_train, **kwds)

def test_PyTorch(files, params=None, specs=None):
    """
    Test function demonstrates workflow of setting up data generator and train
    PyTorch model over given set of files
    """
    from jarray.pytorch import JaggedArrayLinear
    import torch
    if not params:
        params = {}
    if not specs:
        specs = {}
    xfiles = [xfile(f) for f in files]
    gen = DataGenerator(xfiles, params, specs)
    epochs = specs.get('epochs', 10)
    batch_size = specs.get('batch_size', 50)
    shuffle = specs.get('shuffle', True)
    split = specs.get('split', 0.3)
    model = False
    for (x_train, x_mask) in gen:
        if not model:
            input_shape = np.shape(x_train)[-1] # read number of attributes we have
            print("### input data: {}".format(input_shape))
            model = test_PyTorch_model(input_shape)
            print(model)
        print("x_train chunk of {} shape".format(np.shape(x_train)))
        print("x_mask chunk of {} shape".format(np.shape(x_mask)))
        if np.shape(x_train)[0] == 0:
            print("received empty x_train chunk")
            break
        data = np.array([x_train, x_mask])
        preds = model(data).data.numpy()
        print("preds chunk of {} shape".format(np.shape(preds)))


def test(mname):
    input_shape = (10,)
    model = test_Keras_model(input_shape)
    print(type(model))
    model = test_PyTorch_model(input_shape[0])
    print(type(model))
    model = load_model(mname)

if __name__ == '__main__':
    test(sys.argv[1])
