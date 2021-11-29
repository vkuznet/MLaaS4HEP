"""
Basic example of ML model implemented via Keras framework
"""
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense

def model(idim):
    "Simple Keras model for testing purposes"
    ml_model = keras.Sequential([keras.layers.Dense(128, activation='relu',input_shape=(idim,)),
                              keras.layers.Dropout(0.5),
                              keras.layers.Dense(64, activation='relu'),
                              keras.layers.Dropout(0.5),
                              keras.layers.Dense(1, activation='sigmoid')])
    ml_model.compile(optimizer=keras.optimizers.Adam(lr=1e-3), loss=keras.losses.BinaryCrossentropy(),
                  metrics=[keras.metrics.BinaryAccuracy(name='accuracy'), keras.metrics.AUC(name='auc')])
    return ml_model