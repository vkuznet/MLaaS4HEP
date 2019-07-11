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
        Dense(2),
        Activation('softmax'),
    ])
    ml_model.compile(optimizer='adam', \
                  loss='categorical_crossentropy', \
                  metrics=['accuracy'])
    return ml_model
