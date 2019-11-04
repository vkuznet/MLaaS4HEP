"""
Basic example of ML model implemented via Keras framework
"""
from keras.models import Sequential
from keras.layers import Dense, Activation

def model(idim):
    "Simple Keras model for testing purposes"

    ml_model = Sequential([
        Dense(32, input_shape=(idim,)),
        Activation('relu'),
        Dense(2), # use Dense(1) if you have 2 output classes
        Activation('softmax'),
    ])
    ml_model.compile(optimizer='adam', \
                  loss='categorical_crossentropy', \ # use loss='binary_crossentropy' if you have 2 output classes
                  metrics=['accuracy'])
    return ml_model
