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
import time

# numpy modules
import numpy as np

#sklearn modules
from sklearn.model_selection import train_test_split

# keras modules
from tensorflow.keras.utils import to_categorical

# pytorch modules
try:
    import torch
except ImportError:
    torch = None

# MLaaS4HEP modules
from generator import RootDataGenerator, MetaDataGenerator, file_type
from utils import load_code

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
        if self.verbose > 1:
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
            'shuffle': shuffle}

    for data in gen:
        time_ml = time.time()
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
        x_train = np.append(x_train,np.array(y_train).reshape(len(y_train),1),axis=1)

        #create the test set
        train_val, test = train_test_split(x_train, stratify=y_train,test_size=0.2, random_state=21, shuffle=True)
        X_train_val = train_val[:,:-1]
        Y_train_val = train_val[:,-1:]
        X_test = test[:,:-1]
        Y_test = test[:,-1:]
        
        #create the validation set
        train, val = train_test_split(train_val, stratify=Y_train_val, test_size=0.2, random_state=21, shuffle=True)
        X_train=train[:,:-1]
        Y_train=train[:,-1:]
        X_val=val[:,:-1]
        Y_val=val[:,-1:]

        #fit the model
        print(f"\n####Time pre ml: {time.time()-time_ml}")
        time0 = time.time()
        trainer.fit(X_train, Y_train, **kwds, validation_data=(X_val,Y_val))
        print(f"\n####Time for training: {time.time()-time0}\n\n")
    
    if fout and hasattr(trainer, 'save'):
        trainer.save(fout)
