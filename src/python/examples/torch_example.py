from reader import DataReader, xfile
from tfaas import DataGenerator
from jarray.pytorch import JaggedArrayLinear # implementation of LinearLayer for JaggedArray
import torch

params = {} # user provide model parameters, e.g. number of events to read,
specs  = {} # data specs, e.g. jagged array dimentionality
for fin in files:
    fin = xfile(fin)                        # if we're given LFN create xrootd link
    gen = DataGenerator(fin, params, specs) # read data chunk for given file, params/specs
    model = False
    for (x_train, x_mask) in gen:
        if not model:
            input_shape = np.shape(x_train)[-1] # read number of attributes we have
            model = torch.nn.Sequential(
                JaggedArrayLinear(input_shape, 5),
                torch.nn.ReLU(),
                torch.nn.Linear(5, 1),
            )
            print(model)
        if np.shape(x_train)[0] == 0:
            print("received empty x_train chunk")
            break
        data = np.array([x_train, x_mask])
        preds = model(data).data.numpy()
