# MLaaS4HEP worflow recipe

Install anaconda, create a new environment and install all the MLaaS4HEP dependencies

```
conda create -n env
conda activate env
conda config --add channels conda-forge
conda install pyarrow uproot=3.12 numba scikit-learn
conda install keras numpy pandas
```

create a new working directory and download MLaaS4HEP code

```
mkdir work_dir
cd work_dir
git clone https://github.com/vkuznet/MLaaS4HEP.git
```

setup the proper environment

```
export PYTHONPATH=$PYTHONPATH:$PWD/MLaaS4HEP/src/python/
export PATH=$PWD/MLaaS4HEP/bin:$PATH
```

download the following two root files to test MLaaS4HEP with local files

```
wget http://opendata.cern.ch/record/12351/files/GluGluToHToTauTau.root
wget http://opendata.cern.ch/record/12352/files/VBF_HToTauTau.root
```

create a files.txt with the path of the two root files to use

```
GluGluToHToTauTau.root
VBF_HToTauTau.root
```

create a labels.txt with the class labels

```
1
0
```

create a test_model.py file with the definition of the model to use. Choosing a Keras model you can use [keras_model.py](https://github.com/vkuznet/MLaaS4HEP/blob/master/src/python/MLaaS4HEP/keras_model.py) or more in general

```
def model(idim):
   "Simple Keras model for testing purposes"
   ml_model = keras.Sequential(...)
   my_model.compile(...)
   return ml_model
```

eventually you can customize the model and its fit function in [models.py](https://github.com/vkuznet/MLaaS4HEP/blob/121902e323cfce0504f257f28236c041098e7d9c/src/python/MLaaS4HEP/models.py#L39)

```
class MyModel(keras.Model):
   def fit(self, x_train, y_train, **kwds):
       print("my new fit function")
def model(idim):
   "Simple Keras model for testing purposes"
   ml_model = keras.Sequential[...]
   my_model = MyModel(ml_model)
   my_model.compile(...)
   return MyModel(ml_model)
```

create the params.json file with other parameters, e.g.

```
{
    "nevts": 50000,
    "shuffle": true,
    "chunk_size": 10000,
    "epochs": 5,
    "batch_size": 100,
    "identifier": "",
    "branch": “Events",
    "selected_branches": "",
    "exclude_branches": ["run", "event", "luminosityBlock"],
    "hist": "pdfs",
    "redirector": "",
    "verbose": 1
  }
```

run the [workflow.py](https://github.com/vkuznet/MLaaS4HEP/blob/master/src/python/MLaaS4HEP/workflow.py) script

```
./MLaaS4HEP/src/python/MLaaS4HEP/workflow.py --files=files.txt --labels=labels.txt --model=keras_model.py --params=params.json
```

This script performs the following actions:
- read all the ROOT files in chunks to compute the specs file
- perform the training cycle (each time using a new chunk of events)
  - create a new chunk of events taken proportionally from the input ROOT files
    - extract and convert each event in a list of NumPy arrays
    - normalize the events 
    - fix the Jagged Arrays dimension
    - create the masking vector
  - use the chunk to train the ML model provided by the user

## Example of output
Running the [workflow.py](https://github.com/vkuznet/MLaaS4HEP/blob/master/src/python/MLaaS4HEP/workflow.py) script using the [keras_model.py](https://github.com/vkuznet/MLaaS4HEP/blob/master/src/python/MLaaS4HEP/keras_model.py) model and the parameters chosen before, a similar output will be shown
```
./MLaaS4HEP/src/python/MLaaS4HEP/workflow.py --files=files.txt --labels=labels.txt --model=keras_model.py --params=params.json
load keras_model.py <function model at 0x7f6c1281c700> Simple Keras model for testing purposes
DataGenerator: <MLaaS4HEP.generator.RootDataGenerator object at 0x7f6c1281e160> [20/Mar/2021:10:51:24] 1616237484.0
model parameters: {"nevts": 50000, "shuffle": true, "chunk_size": 10000, "epochs": 5, "batch_size": 100, "identifier": "", "branch": "Events", "selected_branches": "", "exclude_branches": ["run", "event", "luminosityBlock"], "hist": "pdfs", "redirector": "", "verbose": 1}
Reading GluGluToHToTauTau.root
Excluded branches: ['run', 'event', 'luminosityBlock']
# 10000 entries, 66 branches, 10.718155860900879 MB, 0.2969963550567627 sec, 36.088509769260966 MB/sec, 33.67044689181043 kHz
# 10000 entries, 66 branches, 10.673151969909668 MB, 0.02723407745361328 sec, 391.9042966698183 MB/sec, 367.1870294498722 kHz
# 10000 entries, 66 branches, 10.758533477783203 MB, 0.023487329483032227 sec, 458.0569061951214 MB/sec, 425.76147310507247 kHz
# 10000 entries, 66 branches, 10.662114143371582 MB, 0.06366372108459473 sec, 167.4755097837281 MB/sec, 157.07533002527853 kHz
# 10000 entries, 66 branches, 10.74631404876709 MB, 0.046987056732177734 sec, 228.7079633444626 MB/sec, 212.82456692274124 kHz
###total time elapsed for reading + specs computing: 1.6907954216003418; number of chunks 5
###total time elapsed for reading: 0.4583611488342285; number of chunks 5

--- first pass: 476963 events, (18-flat, 48-jagged) branches, 1767 attrs
<MLaaS4HEP.reader.RootDataReader object at 0x7f6c1281e2e0> init is complete in 1.6989874839782715 sec
Reading VBF_HToTauTau.root
Excluded branches: ['run', 'event', 'luminosityBlock']
# 10000 entries, 66 branches, 11.964397430419922 MB, 0.5326294898986816 sec, 22.46288960210563 MB/sec, 18.77477719437245 kHz
# 10000 entries, 66 branches, 11.830232620239258 MB, 0.042020320892333984 sec, 281.53598946926456 MB/sec, 237.98009600217875 kHz
# 10000 entries, 66 branches, 11.900396347045898 MB, 0.03052067756652832 sec, 389.9125870028825 MB/sec, 327.64672337965675 kHz
# 10000 entries, 66 branches, 11.864381790161133 MB, 0.05782127380371094 sec, 205.19059871350817 MB/sec, 172.94672604321292 kHz
# 10000 entries, 66 branches, 11.948942184448242 MB, 0.031001806259155273 sec, 385.4272904153625 MB/sec, 322.5618506356177 kHz
###total time elapsed for reading + specs computing: 1.8234634399414062; number of chunks 5
###total time elapsed for reading: 0.6939854621887207; number of chunks 5

--- first pass: 491653 events, (18-flat, 48-jagged) branches, 1785 attrs
<MLaaS4HEP.reader.RootDataReader object at 0x7f6c12836d30> init is complete in 1.8325014114379883 sec
write global-specs.json
load specs from global-specs.json for GluGluToHToTauTau.root
load specs from global-specs.json for VBF_HToTauTau.root
init RootDataGenerator in 3.599780797958374 sec


label 1, file <GluGluToHToTauTau.root>, going to read 4924 events
read chunk [0:4923] from GluGluToHToTauTau.root
# 10000 entries, 66 branches, 10.718155860900879 MB, 0.238877534866333 sec, 44.86883149936372 MB/sec, 41.86245477455898 kHz
total read 4924 evts from GluGluToHToTauTau.root

label 0, file <VBF_HToTauTau.root>, going to read 5076 events
read chunk [4924:9999] from VBF_HToTauTau.root
# 10000 entries, 66 branches, 11.964397430419922 MB, 0.26181483268737793 sec, 45.69793585646885 MB/sec, 38.19493302711607 kHz
total read 5076 evts from VBF_HToTauTau.root


Time for handling a chunk: 9.95328426361084


x_mask chunk of (10000, 1803) shape
x_train chunk of (10000, 1803) shape
y_train chunk of (10000,) shape
2021-03-20 10:51:37.771548: I tensorflow/compiler/jit/xla_cpu_device.cc:41] Not creating XLA devices, tf_xla_enable_xla_devices not set
2021-03-20 10:51:37.771809: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  SSE4.1 SSE4.2 AVX AVX2 AVX512F FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2021-03-20 10:51:37.773096: I tensorflow/core/common_runtime/process_util.cc:146] Creating new thread pool with default inter op setting: 2. Tune using inter_op_parallelism_threads for best performance.
model <tensorflow.python.keras.engine.sequential.Sequential object at 0x7f6c11a42dc0> loss function <tensorflow.python.keras.losses.BinaryCrossentropy object at 0x7f6c11b36d00>
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense (Dense)                (None, 128)               230912    
_________________________________________________________________
dropout (Dropout)            (None, 128)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 64)                8256      
_________________________________________________________________
dropout_1 (Dropout)          (None, 64)                0         
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 65        
=================================================================
Total params: 239,233
Trainable params: 239,233
Non-trainable params: 0
_________________________________________________________________
None

####Time pre ml: 0.43134403228759766
2021-03-20 10:51:38.186190: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:116] None of the MLIR optimization passes are enabled (registered 2)
2021-03-20 10:51:38.186622: I tensorflow/core/platform/profile_utils/cpu_utils.cc:112] CPU Frequency: 2299995000 Hz
Epoch 1/5
64/64 [==============================] - 3s 37ms/step - loss: 0.6881 - accuracy: 0.5581 - auc: 0.5787 - val_loss: 0.6260 - val_accuracy: 0.6744 - val_auc: 0.7317
Epoch 2/5
64/64 [==============================] - 1s 23ms/step - loss: 0.6406 - accuracy: 0.6404 - auc: 0.6863 - val_loss: 0.6032 - val_accuracy: 0.6606 - val_auc: 0.7504
Epoch 3/5
64/64 [==============================] - 1s 10ms/step - loss: 0.6175 - accuracy: 0.6565 - auc: 0.7186 - val_loss: 0.5870 - val_accuracy: 0.7013 - val_auc: 0.7682
Epoch 4/5
64/64 [==============================] - 1s 18ms/step - loss: 0.6008 - accuracy: 0.6775 - auc: 0.7397 - val_loss: 0.5866 - val_accuracy: 0.7000 - val_auc: 0.7675
Epoch 5/5
64/64 [==============================] - 0s 7ms/step - loss: 0.5810 - accuracy: 0.7159 - auc: 0.7690 - val_loss: 0.5686 - val_accuracy: 0.7250 - val_auc: 0.7908

####Time for training: 7.068070650100708


loss :  0.5712021589279175
accuracy :  0.7039999961853027
auc :  0.7821849584579468
label 1, file <GluGluToHToTauTau.root>, going to read 4924 events
read chunk [10000:14923] from GluGluToHToTauTau.root
total read 9848 evts from GluGluToHToTauTau.root

label 0, file <VBF_HToTauTau.root>, going to read 5076 events
read chunk [14924:19999] from VBF_HToTauTau.root
# 10000 entries, 66 branches, 11.830232620239258 MB, 0.03023052215576172 sec, 391.3340483926938 MB/sec, 330.79150761853685 kHz
total read 10152 evts from VBF_HToTauTau.root


Time for handling a chunk: 9.452708959579468


x_mask chunk of (10000, 1803) shape
x_train chunk of (10000, 1803) shape
y_train chunk of (10000,) shape

####Time pre ml: 0.33245253562927246
Epoch 1/5
64/64 [==============================] - 1s 17ms/step - loss: 0.5937 - accuracy: 0.6913 - auc: 0.7506 - val_loss: 0.5671 - val_accuracy: 0.7119 - val_auc: 0.7859
Epoch 2/5
64/64 [==============================] - 1s 8ms/step - loss: 0.5864 - accuracy: 0.6975 - auc: 0.7597 - val_loss: 0.5719 - val_accuracy: 0.6981 - val_auc: 0.7865
Epoch 3/5
64/64 [==============================] - 1s 9ms/step - loss: 0.5781 - accuracy: 0.7083 - auc: 0.7690 - val_loss: 0.5560 - val_accuracy: 0.7344 - val_auc: 0.7957
Epoch 4/5
64/64 [==============================] - 1s 11ms/step - loss: 0.5569 - accuracy: 0.7220 - auc: 0.7906 - val_loss: 0.5522 - val_accuracy: 0.7312 - val_auc: 0.7980
Epoch 5/5
64/64 [==============================] - 0s 7ms/step - loss: 0.5635 - accuracy: 0.7150 - auc: 0.7832 - val_loss: 0.5544 - val_accuracy: 0.7200 - val_auc: 0.7980

####Time for training: 3.443528175354004


loss :  0.5521019697189331
accuracy :  0.7289999723434448
auc :  0.7975825071334839
label 1, file <GluGluToHToTauTau.root>, going to read 4924 events
read chunk [20000:24923] from GluGluToHToTauTau.root
# 10000 entries, 66 branches, 10.673151969909668 MB, 0.020817995071411133 sec, 512.688754767113 MB/sec, 480.353653927643 kHz
total read 14772 evts from GluGluToHToTauTau.root

label 0, file <VBF_HToTauTau.root>, going to read 5076 events
read chunk [24924:29999] from VBF_HToTauTau.root
total read 15228 evts from VBF_HToTauTau.root


Time for handling a chunk: 11.282498121261597


x_mask chunk of (10000, 1803) shape
x_train chunk of (10000, 1803) shape
y_train chunk of (10000,) shape

####Time pre ml: 0.37798213958740234
Epoch 1/5
64/64 [==============================] - 1s 8ms/step - loss: 0.5703 - accuracy: 0.7136 - auc: 0.7789 - val_loss: 0.5577 - val_accuracy: 0.7206 - val_auc: 0.7969
Epoch 2/5
64/64 [==============================] - 0s 4ms/step - loss: 0.5618 - accuracy: 0.7177 - auc: 0.7867 - val_loss: 0.5443 - val_accuracy: 0.7225 - val_auc: 0.8050
Epoch 3/5
64/64 [==============================] - 0s 4ms/step - loss: 0.5597 - accuracy: 0.7211 - auc: 0.7891 - val_loss: 0.5426 - val_accuracy: 0.7312 - val_auc: 0.8065
Epoch 4/5
64/64 [==============================] - 0s 4ms/step - loss: 0.5465 - accuracy: 0.7344 - auc: 0.7996 - val_loss: 0.5411 - val_accuracy: 0.7312 - val_auc: 0.8054
Epoch 5/5
64/64 [==============================] - 1s 17ms/step - loss: 0.5398 - accuracy: 0.7356 - auc: 0.8072 - val_loss: 0.5405 - val_accuracy: 0.7306 - val_auc: 0.8062

####Time for training: 2.6567909717559814


loss :  0.5259018540382385
accuracy :  0.737500011920929
auc :  0.8183405995368958
label 1, file <GluGluToHToTauTau.root>, going to read 4924 events
read chunk [30000:34923] from GluGluToHToTauTau.root
total read 19696 evts from GluGluToHToTauTau.root

label 0, file <VBF_HToTauTau.root>, going to read 5076 events
read chunk [34924:39999] from VBF_HToTauTau.root
# 10000 entries, 66 branches, 11.900396347045898 MB, 0.027304887771606445 sec, 435.83392272429603 MB/sec, 366.2347958960925 kHz
total read 20304 evts from VBF_HToTauTau.root


Time for handling a chunk: 9.558260440826416


x_mask chunk of (10000, 1803) shape
x_train chunk of (10000, 1803) shape
y_train chunk of (10000,) shape

####Time pre ml: 0.3266947269439697
Epoch 1/5
64/64 [==============================] - 0s 7ms/step - loss: 0.5356 - accuracy: 0.7384 - auc: 0.8104 - val_loss: 0.5320 - val_accuracy: 0.7450 - val_auc: 0.8172
Epoch 2/5
64/64 [==============================] - 0s 5ms/step - loss: 0.5313 - accuracy: 0.7437 - auc: 0.8152 - val_loss: 0.5399 - val_accuracy: 0.7362 - val_auc: 0.8052
Epoch 3/5
64/64 [==============================] - 0s 8ms/step - loss: 0.5261 - accuracy: 0.7469 - auc: 0.8181 - val_loss: 0.5268 - val_accuracy: 0.7331 - val_auc: 0.8213
Epoch 4/5
64/64 [==============================] - 1s 18ms/step - loss: 0.5224 - accuracy: 0.7530 - auc: 0.8217 - val_loss: 0.5335 - val_accuracy: 0.7387 - val_auc: 0.8132
Epoch 5/5
64/64 [==============================] - 0s 4ms/step - loss: 0.5219 - accuracy: 0.7566 - auc: 0.8233 - val_loss: 0.5262 - val_accuracy: 0.7412 - val_auc: 0.8163

####Time for training: 2.8685789108276367


loss :  0.531890332698822
accuracy :  0.7390000224113464
auc :  0.8135925531387329
label 1, file <GluGluToHToTauTau.root>, going to read 4924 events
read chunk [40000:44923] from GluGluToHToTauTau.root
# 10000 entries, 66 branches, 10.758533477783203 MB, 0.02563333511352539 sec, 419.7086891009543 MB/sec, 390.1170080175605 kHz
total read 24620 evts from GluGluToHToTauTau.root

label 0, file <VBF_HToTauTau.root>, going to read 5075 events
read chunk [44924:49998] from VBF_HToTauTau.root
total read 25379 evts from VBF_HToTauTau.root


Time for handling a chunk: 9.724456787109375


x_mask chunk of (9999, 1803) shape
x_train chunk of (9999, 1803) shape
y_train chunk of (9999,) shape

####Time pre ml: 0.3025078773498535
Epoch 1/5
64/64 [==============================] - 2s 21ms/step - loss: 0.5586 - accuracy: 0.7262 - auc: 0.7912 - val_loss: 0.5249 - val_accuracy: 0.7600 - val_auc: 0.8247
Epoch 2/5
64/64 [==============================] - 1s 19ms/step - loss: 0.5461 - accuracy: 0.7368 - auc: 0.8027 - val_loss: 0.5183 - val_accuracy: 0.7531 - val_auc: 0.8283
Epoch 3/5
64/64 [==============================] - 1s 13ms/step - loss: 0.5419 - accuracy: 0.7373 - auc: 0.8038 - val_loss: 0.5184 - val_accuracy: 0.7519 - val_auc: 0.8286
Epoch 4/5
64/64 [==============================] - 1s 9ms/step - loss: 0.5334 - accuracy: 0.7443 - auc: 0.8128 - val_loss: 0.5100 - val_accuracy: 0.7544 - val_auc: 0.8314
Epoch 5/5
64/64 [==============================] - 1s 9ms/step - loss: 0.5303 - accuracy: 0.7431 - auc: 0.8147 - val_loss: 0.5092 - val_accuracy: 0.7631 - val_auc: 0.8334

####Time for training: 5.780688524246216
```

The plot of the AUC score will be something like this

![AUC_plot](https://github.com/lgiommi/MLaaS4HEP/blob/production/doc/AUC_open_data.jpg)
