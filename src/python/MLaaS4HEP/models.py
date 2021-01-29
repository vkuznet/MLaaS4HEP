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

    accuracy_train=[]
    accuracy_test=[]
    log_loss_train=[]
    log_loss_test=[]
    auc_train=[]
    auc_test=[]
    from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
    for data in gen:
        time_ml=time.time()
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
            #import pandas as pd
            #pd.DataFrame(x_train).to_csv('x_train.csv', header=None, index=None, mode='a')
            #pd.DataFrame(y_train).to_csv('y_train.csv', header=None, index=None, mode='a')
            #print("x_mask chunk of {} shape".format(np.shape(x_mask)))
        #print("x_train chunk of {} shape".format(np.shape(x_train)))
        #print("y_train chunk of {} shape".format(np.shape(y_train)))
        if not trainer:
            idim = np.shape(x_train)[-1] # read number of attributes we have
            model = model(idim)
            print("model", model, "loss function", model.loss)
            trainer = Trainer(model, verbose=params.get('verbose', 0))
        # convert y_train to categorical array
        #if model.loss == 'categorical_crossentropy':
        #    y_train = to_categorical(y_train)
    #print(f"Total time for preparing data for training: {time.time()-time_start}")
        x_train=np.append(x_train,np.array(y_train).reshape(len(y_train),1),axis=1)
        train_val, test = train_test_split(x_train, stratify=y_train,test_size=0.2, random_state=21, shuffle=True)
        X_train_val=train_val[:,:-1]
        Y_train_val=train_val[:,-1:]
        X_test=test[:,:-1]
        Y_test=test[:,-1:]
        
        #create the validation set
        train, val = train_test_split(train_val, stratify=Y_train_val, test_size=0.2, random_state=21, shuffle=True)
        X_train=train[:,:-1]
        Y_train=train[:,-1:]
        X_val=val[:,:-1]
        Y_val=val[:,-1:]
        #pd.DataFrame(X_test).to_csv('x_test_prova.csv', header=None, index=None, mode='a')
        #pd.DataFrame(Y_test).to_csv('y_test_prova.csv', header=None, index=None, mode='a')
        
        #pd.DataFrame(X_train).to_csv('x_train_new.csv', header=None, index=None, mode='a')
        #pd.DataFrame(Y_train).to_csv('y_train_new.csv', header=None, index=None, mode='a')
        #pd.DataFrame(X_val).to_csv('x_val_new.csv', header=None, index=None, mode='a')
        #pd.DataFrame(Y_val).to_csv('y_val_new.csv', header=None, index=None, mode='a')
        #pd.DataFrame(X_test).to_csv('x_test_new.csv', header=None, index=None, mode='a')
        #pd.DataFrame(Y_test).to_csv('y_test_new.csv', header=None, index=None, mode='a')
        #import tensorflow as tf
        #dizio={'test_loss':[], 'test_accuracy':[], 'test_auc':[]}
        #class TestCallback(tf.keras.callbacks.Callback):
        #   def on_epoch_end(self, epoch, logs=None):
        #       loss, acc, auc= self.model.evaluate(X_test, Y_test, verbose=0)
        #       dizio['test_loss'].append(loss)
        #       dizio['test_accuracy'].append(acc)
        #       dizio['test_auc'].append(auc)
        #       print('\nTesting loss: {}, acc: {}, auc: {}\n'.format(loss, acc, auc))
        
        #fit the model
        print(f"\n####Time pre ml: {time.time()-time_ml}")
        time0=time.time()
        model.fit(X_train, Y_train, **kwds, validation_data=(X_val,Y_val))
        print(f"\n####Time for training: {time.time()-time0}")
    #X_test = pd.read_csv('x_test_prova.csv').values
    #Y_test = pd.read_csv('y_test_prova.csv').values
    #results = model.evaluate(X_test, Y_test, verbose=0)
    #print(results)
        '''
        print(dizio)

        import json,codecs
        def saveHist(history):

            new_hist = {}
            for key in list(history.history.keys()):
                print(f"key: {key}; type: {type(history.history[key])}")
                if type(history.history[key]) == np.ndarray:
                    new_hist[key] = history.history[key].tolist()
                elif type(history.history[key]) == list:
                    if  type(history.history[key][0]) == np.float64:
                        new_hist[key] = list(map(float, history.history[key]))
                    if  type(history.history[key][0]) == np.float32:
                        new_hist[key] = list(map(float, np.float64(history.history[key])))

            print(new_hist)
            with codecs.open("history_new.json", 'w', encoding='utf-8') as file:
                json.dump(new_hist, file, separators=(',', ':'), sort_keys=True, indent=4)
        
        saveHist(result)

    #if fout and hasattr(trainer, 'save'):
        model.save('model_new.h5')

        #y_pred_train_val=model.predict_classes(X_train_val)  #these two is for keras model
        #y_pred_test=model.predict_classes(X_test)
        #y_pred_train_val_proba=model.predict_proba(X_train_val)
        #y_pred_test_proba=model.predict_proba(X_test)
        #accuracy_train.append(accuracy_score(Y_train_val, y_pred_train_val))
        #accuracy_test.append(accuracy_score(Y_test, y_pred_test))
        #log_loss_train.append(log_loss(Y_train_val, y_pred_train_val_proba))
        #log_loss_test.append(log_loss(Y_test, y_pred_test_proba))
        #time_end=time.time()
        #print(f"\nTime for loading a chunk of data and training the model: {time_end-time_start} \n")
        

        baseline_results = model.evaluate(X_train_val, Y_train_val, verbose=0)
        for name, value in zip(model.metrics_names, baseline_results):
            if name == 'loss':
                log_loss_train.append(value)
            if name == 'auc':
                auc_train.append(value)
            if name == 'accuracy':
                accuracy_train.append(value)

        baseline_results = model.evaluate(X_test, Y_test, verbose=0)
        for name, value in zip(model.metrics_names, baseline_results):
            if name == 'loss':
                log_loss_test.append(value)
            if name == 'auc':
                auc_test.append(value)
            if name == 'accuracy':
                accuracy_test.append(value)
            print(name, ': ', value)

    print(f"Accuracy_train: {accuracy_train}, log_loss_train: {log_loss_train}, Accuracy_test: {accuracy_test}, log_loss_test: {log_loss_test}")
    print(f"Auc_train: {auc_train}, Auc_test: {auc_test}")

    #if fout and hasattr(trainer, 'save'):
    #    trainer.save(fout)



        #from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
        #model.partial_fit(x_train, y_train, classes=[0, 1])
        #model.fit(x_train, y_train)
        #y_predicted = model.predict(x_train)
        #print(y_train)
        #print(f"Accuracy: {accuracy_score(y_train, y_predicted)}, log_loss: {log_loss(y_train, y_predicted)}")
    '''
    '''
        accuracy_train=[]
        accuracy_test=[]
        log_loss_train=[]
        log_loss_test=[]
        from sklearn.model_selection import train_test_split
        x_train=np.append(x_train,np.array(y_train).reshape(len(y_train),1),axis=1)
        train, test = train_test_split(x_train, stratify=y_train,test_size=0.2, random_state=21, shuffle=True)
        X_train=train[:,:-1]
        Y_train=train[:,-1:]
        X_test=test[:,:-1]
        Y_test=test[:,-1:]
        #model.partial_fit(X_train, Y_train.reshape(len(Y_train),), classes=[0, 1])
        result=model.fit(X_train, Y_train, **kwds)
        y_pred_train=model.predict_classes(X_train)  #these two is for keras model
        y_pred_test=model.predict_classes(X_test)
        #y_pred_train=model.predict(X_train)  #these two are for sgd
        #y_pred_test=model.predict(X_test)
        #print (y_pred_train)
        y_pred_train_proba=model.predict_proba(X_train)
        y_pred_test_proba=model.predict_proba(X_test)
        accuracy_train.append(accuracy_score(Y_train, y_pred_train))
        accuracy_test.append(accuracy_score(Y_test, y_pred_test))
        #log_loss_train.append(log_loss(Y_train, y_pred_train_proba))
        #log_loss_test.append(log_loss(Y_test, y_pred_test_proba))
        #print(f"Accuracy_train: {accuracy_score(Y_train, y_pred_train)}, log_loss_train: {log_loss(Y_train, y_pred_train_proba)}, Accuracy_test: {accuracy_score(Y_test, y_pred_test)}, log_loss_test: {log_loss(Y_test, y_pred_test_proba)}")
        #if i==37:
        print(f"accuracy_train: {accuracy_train}")
        print(f"accuracy_test: {accuracy_test}")
            #print(f"log_loss_train: {log_loss_train}")
            #print(f"log_loss_test: {log_loss_test}")'''

    '''from sklearn.model_selection import train_test_split
        x_train=np.append(x_train,np.array(y_train).reshape(len(y_train),1),axis=1)
        train, test = train_test_split(x_train, stratify=y_train,test_size=0.2, random_state=17)
        X_train=train[:,:-1]
        Y_train=train[:,-1:]
        X_test=test[:,:-1]
        Y_test=test[:,-1:]
        import pandas as pd
        pd.DataFrame(X_train).to_csv('x_train.csv', header=None, index=None, mode='a')
        pd.DataFrame(Y_train).to_csv('y_train.csv', header=None, index=None, mode='a')
        pd.DataFrame(X_test).to_csv('x_test.csv', header=None, index=None, mode='a')
        pd.DataFrame(Y_test).to_csv('y_test.csv', header=None, index=None, mode='a')
        result=model.fit(X_train,Y_train, epochs=5, batch_size=100, verbose=1, validation_split=0.3)
        y_pred_train=model.predict_classes(X_train)
        y_pred_test=model.predict_classes(X_test)
        from sklearn.metrics import confusion_matrix
        print(confusion_matrix(Y_test, y_pred_test))
        print(model.evaluate(X_test, Y_test, verbose=0))

        y_pred_train=model.predict(X_train)
        y_pred_test=model.predict(X_test)

        import matplotlib as mpl
        import matplotlib.pyplot as plt
        from sklearn import metrics
        mpl.rcParams['figure.figsize'] = (12, 10)
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        def plot_roc(name, labels, predictions, **kwargs):
            fp, tp, _ = metrics.roc_curve(labels, predictions)

            plt.plot(100*fp, 100*tp, label=name, linewidth=2, **kwargs)
            plt.xlabel('False positives [%]')
            plt.ylabel('True positives [%]')
            plt.xlim([-0.5,100.5])
            plt.ylim([-0.5,100.5])
            plt.grid(True)
            ax = plt.gca()
            ax.set_aspect('equal')
            plt.legend(loc='lower right')
            plt.savefig("plot_roc.png")
            print (metrics.auc(fp,tp))

        plot_roc("Train Baseline", Y_train[:,0], y_pred_train[:,0], color=colors[0])
        plot_roc("Test Baseline", Y_test[:,0], y_pred_test[:,0], color=colors[0], linestyle='--')

        def plot_metrics(history):
            metrics =  ['loss', 'auc']
            for n, metric in enumerate(metrics):
                name = metric.replace("_"," ").capitalize()
                plt.subplot(2,2,n+1)
                plt.plot(history.epoch,  history.history[metric], color=colors[0], label='Train')
                plt.plot(history.epoch, history.history['val_'+metric], color=colors[0], linestyle="--", label='Val')
                plt.xlabel('Epoch')
                plt.xticks(np.arange(0, 5, 1), [1,2,3,4,5])
                plt.ylabel(name)
                if metric == 'loss':
                    plt.ylim([0.20, plt.ylim()[1]])
                elif metric == 'auc':
                    plt.ylim([0.7,1])
                else:
                    plt.ylim([0,1])

                plt.legend()
                plt.savefig("plot_loss_auc_.png")
    
        plot_metrics(result)

        import json
        #with open('file.json', 'w', encoding='utf-8') as f:
        #    json.dump(str(result.history, f))
        #hist_df = pd.DataFrame(result.history)
        #hist_csv_file = 'history.csv'
        #with open(hist_csv_file, mode='w') as f:
        #    hist_df.to_csv(f)
        #hist_json_file = 'history.json' 
        #with open(hist_json_file, mode='w') as f:
        #    hist_df.to_json(f)
        
        import json,codecs
        def saveHist(history):

            new_hist = {}
            for key in list(history.history.keys()):
                print(f"key: {key}; type: {type(history.history[key])}")
                if type(history.history[key]) == np.ndarray:
                    new_hist[key] = history.history[key].tolist()
                elif type(history.history[key]) == list:
                    if  type(history.history[key][0]) == np.float64:
                        new_hist[key] = list(map(float, history.history[key]))
                    if  type(history.history[key][0]) == np.float32:
                        new_hist[key] = list(map(float, np.float64(history.history[key])))

            print(new_hist)
            with codecs.open("history_new.json", 'w', encoding='utf-8') as file:
                json.dump(new_hist, file, separators=(',', ':'), sort_keys=True, indent=4)
        
        saveHist(result)

    #if fout and hasattr(trainer, 'save'):
        model.save('model_new.h5')'''