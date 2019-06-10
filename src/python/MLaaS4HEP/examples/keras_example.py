from reader import DataReader, xfile
from tfaas import DataGenerator

class Trainer(object):
    def __init__(self, model, verbose=0):
        self.model = model
    def fit(self, data, y_train, **kwds):
        xdf, mask = data[0], data[1]
        xdf[np.isnan(mask)] = 0 # case values in data vector according to mask
        self.model.fit(xdf, y_train, verbose=self.verbose, **kwds)
    def predict(self):
        pass # NotImplementedYet

def testModel(input_shape):
    from keras.models import Sequential
    from keras.layers import Dense, Activation
    model = Sequential([Dense(32, input_shape=input_shape), Activation('relu'),
        Dense(2), Activation('softmax')])
    model.compile(optimizer='adam', loss='categorical_crossentropy',  metrics=['accuracy'])
    return model

def testKeras(files, params, specs):
    from keras.utils import to_categorical
    for fin in files:
        fin = xfile(fin)
        gen = DataGenerator(fin, params, specs)
        epochs = specs.get('epochs', 10)
        batch_size = specs.get('batch_size', 50)
        shuffle = specs.get('shuffle', True)
        split = specs.get('split', 0.3)
        trainer = False
        for data in gen:
            x_train = np.array(data[0])
            if not trainer:
                input_shape = (np.shape(x_train)[-1],) # read number of attributes we have
                trainer = Trainer(testModel(input_shape), verbose=params.get('verbose', 0))
            if np.shape(x_train)[0] == 0:
                print("received empty x_train chunk")
                break
            y_train = np.random.randint(2, size=np.shape(x_train)[0]) # dummy vector
            y_train = to_categorical(y_train) # convert labesl to categorical values
            kwds = {'epochs':epochs, 'batch_size': batch_size, 'shuffle': shuffle, 'validation_split': split}
            trainer.fit(data, y_train, **kwds)
