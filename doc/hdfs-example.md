### Predict activity of Rucio data placement
Let's say our objective is to process rucio data on HDFS and build ML model to
predict activity based on some set of attributes. In our example we'll
use MLaas4HEP package and Rucio data on HDFS.

Here we provide all steps (from setup to ML training) that you can
use as an example for future use-cases:

#### setup proper environment
Here we'll use anaconda as an example, but it is not a requirement, e.g. there
are other methods to install all dependencies).
```
# download anaconda shell script and install it on your node

# create new environment
conda create --name=mlaas python=3.7

# activate new environment
conda activate mlaas

# install required dependencies
conda install keras pyarrow
```

Next, we get MLaaS4HEP code
```
git clone git@github.com:vkuznet/MLaaS4HEP.git
cd MLaaS4HEP
```

And, create a project (I'll use rucio as an example)
```
mkdir rucio
# copy example keras model and preprocessing code
cp src/python/MLaaS4HEP/ex_keras.py rucio/rucio_model.py
cp src/python/MLaaS4HEP/ex_preproc.py rucio/rucio_preroc.py
cd rucio
```

Then, we decide which data to use from HDFS and inspect them, e.g.
```
# we'll use rucio data
hadoop fs -ls /project/monitoring/archive/rucio

# download one file to get idea about its structure
hadoop fs -get /project/monitoring/archive/rucio/raw/events/2019/08/15/part-00000-ddb030f3-fd5d-4ecd-9497-b695919417d7-c000.json.gz

# look-up at data structure to decide which attributes to use
zcat part-00000-ddb030f3-fd5d-4ecd-9497-b695919417d7-c000.json.gz | head -1
```

Next, we prepare list of files to work with
```
hadoop fs -ls /project/monitoring/archive/rucio/raw/events/2019/08/15 | awk '{print "hdfs://"$8""}' > rucio_events.txt
```

And, we may edit model/preproc code according to your ML studies, see my code examples below

Finally, we run workflow to train the model
```
workflow --files=rucio_events.txt --labels=activity --model=rucio_model.py --preproc=rucio_preproc.py --fout=model.pb --params=params.json
```

This workflow runs over HDFS data, preprocess them according to
`rucio_preproc.py` code, builds binary classifier (Neural Net keras model) based
on chosen attributes and write out final model into model.pb file.

Here is a list of files and their content I used in above workflow run:

- params.json (configuration file)
```
{
    "nevts": 3000,
    "shuffle": true,
    "chunk_size": 1000,
    "epochs": 5,
    "batch_size": 256,
    "hist": "pdfs",
    "verbose": 1
}
```

- rucio_events.txt our input files
```
hdfs:///project/monitoring/archive/rucio/raw/events/2019/08/15/part-00000-ddb030f3-fd5d-4ecd-9497-b695919417d7-c000.json.gz
hdfs:///project/monitoring/archive/rucio/raw/events/2019/08/15/part-00001-ddb030f3-fd5d-4ecd-9497-b695919417d7-c000.json.gz
...
```

- `rucio_preproc.py` is our preprocessing function. Here I specified list of attributes
  which will be used in our model and decided to build a model which will predict
  `activity base` on `event_type, bytes, file_size and tf_created_at`.
  That's why I selected all of them from given record and return back to workflow.
  I also converted categorical values into numerical ones based on simple rules.
```
def preprocessing(rec):
    "Simple preprocessing function"
    # for example our JSON record has the following structure:
    # {'data': {... payload record ..}} and we want to extract the payload from it
    attrs = ['activity', 'event_type', 'bytes', 'file_size', 'tf_created_at']
    doc = {}
    if isinstance(rec, dict) and 'data' in rec:
        for key, val in rec['data'].items():
            if key not in attrs:
                continue
            if key == 'event_type':
                doc[key] = 1 if val == 'transfer-submitted' else 0
            elif key == 'activity':
                doc[key] = 1 if val == 'Staging' else 0
            else:
                doc[key] = val
    if 'activity' not in doc:
        doc['activity'] = 0 # this is our label, so it should be present
    return doc
```

- rucio_model.py ML model for our ML case studies, here we create simple NN
  binary classifier which will predict activity based on pre-selected attributes.
```
from keras.models import Sequential
from keras.layers import Dense, Activation

def model(idim):
    "Simple Keras model for testing purposes"

    ml_model = Sequential([
        Dense(32, input_shape=(idim,)),
        Activation('relu'),
        Dense(1),
        Activation('softmax'),
    ])
    ml_model.compile(optimizer='adam', \
                  loss='binary_crossentropy', \
                  metrics=['accuracy'])
    return ml_model
```
For completeness here is full output of training phase:

```
workflow --files=rucio_events.txt --labels=activity --model=rucio_model.py --preproc=rucio_preproc.py --fout=model.pb --params=params.json
Using TensorFlow backend.
load rucio_model.py <function model at 0x7f1e0af0a7a0> Simple Keras model for testing purposes
load rucio_preproc.py <function preprocessing at 0x7f1e0af0a830> Simple preprocessing function
Generator: <MLaaS4HEP.generator.MetaDataGenerator object at 0x7f1e0af0d590> [16/Aug/2019:19:13:46] 1565975626.0
model parameters: {"nevts": 3000, "shuffle": true, "chunk_size": 1000, "epochs": 5, "batch_size": 256, "hist": "pdfs", "verbose": 1}
init JsonReader with <MLaaS4HEP.reader.HDFSJSONReader object at 0x7f1e0af0dc10>
init JsonReader with <MLaaS4HEP.reader.HDFSJSONReader object at 0x7f1e0af0dc50>
init JsonReader with <MLaaS4HEP.reader.HDFSJSONReader object at 0x7f1e0af0dd50>
init JsonReader with <MLaaS4HEP.reader.HDFSJSONReader object at 0x7f1e0af0ddd0>
init JsonReader with <MLaaS4HEP.reader.HDFSJSONReader object at 0x7f1e0af0de90>
init JsonReader with <MLaaS4HEP.reader.HDFSJSONReader object at 0x7f1e0af0df10>
init JsonReader with <MLaaS4HEP.reader.HDFSJSONReader object at 0x7f1e0af0db50>
init JsonReader with <MLaaS4HEP.reader.HDFSJSONReader object at 0x7f1e0af0df90>
init JsonReader with <MLaaS4HEP.reader.HDFSJSONReader object at 0x7f1e0af16050>
init MetaDataGenerator in 0.00044608116149902344 sec
available readers
hdfs:///project/monitoring/archive/rucio/raw/events/2019/08/15/part-00000-ddb030f3-fd5d-4ecd-9497-b695919417d7-c000.json.gz <MLaaS4HEP.reader.JsonReader object at 0x7f1e0af0db10>
hdfs:///project/monitoring/archive/rucio/raw/events/2019/08/15/part-00001-ddb030f3-fd5d-4ecd-9497-b695919417d7-c000.json.gz <MLaaS4HEP.reader.JsonReader object at 0x7f1e0af0dc90>
hdfs:///project/monitoring/archive/rucio/raw/events/2019/08/15/part-00002-ddb030f3-fd5d-4ecd-9497-b695919417d7-c000.json.gz <MLaaS4HEP.reader.JsonReader object at 0x7f1e0af0dd10>
hdfs:///project/monitoring/archive/rucio/raw/events/2019/08/15/part-00003-ddb030f3-fd5d-4ecd-9497-b695919417d7-c000.json.gz <MLaaS4HEP.reader.JsonReader object at 0x7f1e0af0dd90>
hdfs:///project/monitoring/archive/rucio/raw/events/2019/08/15/part-00004-ddb030f3-fd5d-4ecd-9497-b695919417d7-c000.json.gz <MLaaS4HEP.reader.JsonReader object at 0x7f1e0af0de10>
hdfs:///project/monitoring/archive/rucio/raw/events/2019/08/15/part-00005-ddb030f3-fd5d-4ecd-9497-b695919417d7-c000.json.gz <MLaaS4HEP.reader.JsonReader object at 0x7f1e0af0ded0>
hdfs:///project/monitoring/archive/rucio/raw/events/2019/08/15/part-00006-ddb030f3-fd5d-4ecd-9497-b695919417d7-c000.json.gz <MLaaS4HEP.reader.JsonReader object at 0x7f1e0af0df50>
hdfs:///project/monitoring/archive/rucio/raw/events/2019/08/15/part-00007-ddb030f3-fd5d-4ecd-9497-b695919417d7-c000.json.gz <MLaaS4HEP.reader.JsonReader object at 0x7f1e0af0d810>
hdfs:///project/monitoring/archive/rucio/raw/events/2019/08/15/part-00008-ddb030f3-fd5d-4ecd-9497-b695919417d7-c000.json.gz <MLaaS4HEP.reader.JsonReader object at 0x7f1e0af0dfd0>

read chunk [0:1000] from hdfs:///project/monitoring/archive/rucio/raw/events/2019/08/15/part-00000-ddb030f3-fd5d-4ecd-9497-b695919417d7-c000.json.gz label activity
19/08/16 19:13:47 WARN util.NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable
19/08/16 19:13:48 WARN shortcircuit.DomainSocketFactory: The short-circuit local reads feature cannot be used because libhadoop cannot be loaded.

total read 1000 evts from hdfs:///project/monitoring/archive/rucio/raw/events/2019/08/15/part-00000-ddb030f3-fd5d-4ecd-9497-b695919417d7-c000.json.gz
return shapes: data=(3000, 4) labels=(3000,)
x_train chunk of (3000, 4) shape
y_train chunk of (3000,) shape
y_train chunk of (3000,) shape
WARNING: Logging before flag parsing goes to stderr.
W0816 19:14:03.861870 139767866615616 deprecation_wrapper.py:119] From /wma/vk/Anaconda/envs/mlaas/lib/python3.7/site-packages/keras/backend/tensorflow_backend.py:74: The name tf.get_default_graph is deprecated. Please use tf.compat.v1.get_W0816 19:14:03.884509 139767866615616 deprecation_wrapper.py:119] From /wma/vk/Anaconda/envs/mlaas/lib/python3.7/site-packages/keras/backend/tensorflow_backend.py:4138: The name tf.random_uniform is deprecated. Please use tf.random.uniform instead.

W0816 19:14:03.915873 139767866615616 deprecation_wrapper.py:119] From /wma/vk/Anaconda/envs/mlaas/lib/python3.7/site-packages/keras/optimizers.py:790: The name tf.train.Optimizer is deprecated. Please use tf.compat.v1.train.Optimizer instead.

W0816 19:14:03.938851 139767866615616 deprecation_wrapper.py:119] From /wma/vk/Anaconda/envs/mlaas/lib/python3.7/site-packages/keras/backend/tensorflow_backend.py:3376: The name tf.log is deprecated. Please use tf.math.log instead.

W0816 19:14:03.943712 139767866615616 deprecation.py:323] From /wma/vk/Anaconda/envs/mlaas/lib/python3.7/site-packages/tensorflow/python/ops/nn_impl.py:180: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
model <keras.engine.sequential.Sequential object at 0x7f1e0af16950> loss function binary_crossentropy
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense_1 (Dense)              (None, 32)                160
_________________________________________________________________
activation_1 (Activation)    (None, 32)                0
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 33
_________________________________________________________________
activation_2 (Activation)    (None, 1)                 0
=================================================================
Total params: 193
Trainable params: 193
Non-trainable params: 0
_________________________________________________________________
None
Perform fit on (3000, 4) data with {'epochs': 5, 'batch_size': 256, 'shuffle': True, 'validation_split': 0.3}
W0816 19:14:04.139932 139767866615616 deprecation_wrapper.py:119] From /wma/vk/Anaconda/envs/mlaas/lib/python3.7/site-packages/keras/backend/tensorflow_backend.py:986: The name tf.assign_add is deprecated. Please use tf.compat.v1.assign_add instead.

Train on 2100 samples, validate on 900 samples
Epoch 1/5
2019-08-16 19:14:04.272039: I tensorflow/core/platform/cpu_feature_guard.cc:145] This TensorFlow binary is optimized with Intel(R) MKL-DNN to use the following CPU instructions in performance critical operations:  SSE4.1 SSE4.2 AVX AVX2 FMA
To enable them in non-MKL-DNN operations, rebuild TensorFlow with the appropriate compiler flags.
2019-08-16 19:14:04.287480: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2194915000 Hz
2019-08-16 19:14:04.289193: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x564538942ed0 executing computations on platform Host. Devices:
2019-08-16 19:14:04.289278: I tensorflow/compiler/xla/service/service.cc:175]   StreamExecutor device (0): <undefined>, <undefined>
OMP: Info #212: KMP_AFFINITY: decoding x2APIC ids.
OMP: Info #210: KMP_AFFINITY: Affinity capable, using global cpuid leaf 11 info
OMP: Info #154: KMP_AFFINITY: Initial OS proc set respected: 0-15
OMP: Info #156: KMP_AFFINITY: 16 available OS procs
OMP: Info #157: KMP_AFFINITY: Uniform topology
OMP: Info #179: KMP_AFFINITY: 16 packages x 1 cores/pkg x 1 threads/core (16 total cores)
OMP: Info #214: KMP_AFFINITY: OS proc to physical thread map:
OMP: Info #171: KMP_AFFINITY: OS proc 0 maps to package 0
OMP: Info #171: KMP_AFFINITY: OS proc 1 maps to package 1
OMP: Info #171: KMP_AFFINITY: OS proc 2 maps to package 2
OMP: Info #171: KMP_AFFINITY: OS proc 3 maps to package 3
OMP: Info #171: KMP_AFFINITY: OS proc 4 maps to package 4
OMP: Info #171: KMP_AFFINITY: OS proc 5 maps to package 5
OMP: Info #171: KMP_AFFINITY: OS proc 6 maps to package 6
OMP: Info #171: KMP_AFFINITY: OS proc 7 maps to package 7
OMP: Info #171: KMP_AFFINITY: OS proc 8 maps to package 8
OMP: Info #171: KMP_AFFINITY: OS proc 9 maps to package 9
OMP: Info #171: KMP_AFFINITY: OS proc 10 maps to package 10
OMP: Info #171: KMP_AFFINITY: OS proc 11 maps to package 11
OMP: Info #171: KMP_AFFINITY: OS proc 12 maps to package 12
OMP: Info #171: KMP_AFFINITY: OS proc 13 maps to package 13
OMP: Info #171: KMP_AFFINITY: OS proc 14 maps to package 14
OMP: Info #171: KMP_AFFINITY: OS proc 15 maps to package 15
OMP: Info #250: KMP_AFFINITY: pid 19758 tid 19758 thread 0 bound to OS proc set 0
2019-08-16 19:14:04.292140: I tensorflow/core/common_runtime/process_util.cc:115] Creating new thread pool with default inter op setting: 2. Tune using inter_op_parallelism_threads for best performance.
2019-08-16 19:14:04.415013: W tensorflow/compiler/jit/mark_for_compilation_pass.cc:1412] (One-time warning): Not using XLA:CPU for cluster because envvar TF_XLA_FLAGS=--tf_xla_cpu_global_jit was not set.  If you want XLA:CPU, either set that envvar, or use experimental_jit_scope to enable XLA:CPU.  To confirm that XLA is active, pass --vmodule=xla_compilation_cache=1 (as a proper command-line flag, not via TF_XLA_FLAGS) or set the envvar XLA_FLAGS=--xla_hlo_profile.
OMP: Info #250: KMP_AFFINITY: pid 19758 tid 19924 thread 1 bound to OS proc set 1
OMP: Info #250: KMP_AFFINITY: pid 19758 tid 19928 thread 5 bound to OS proc set 5
OMP: Info #250: KMP_AFFINITY: pid 19758 tid 19929 thread 6 bound to OS proc set 6
OMP: Info #250: KMP_AFFINITY: pid 19758 tid 19930 thread 7 bound to OS proc set 7
OMP: Info #250: KMP_AFFINITY: pid 19758 tid 19931 thread 8 bound to OS proc set 8
OMP: Info #250: KMP_AFFINITY: pid 19758 tid 19932 thread 9 bound to OS proc set 9
OMP: Info #250: KMP_AFFINITY: pid 19758 tid 19933 thread 10 bound to OS proc set 10
OMP: Info #250: KMP_AFFINITY: pid 19758 tid 19939 thread 16 bound to OS proc set 0
OMP: Info #250: KMP_AFFINITY: pid 19758 tid 19923 thread 17 bound to OS proc set 1
OMP: Info #250: KMP_AFFINITY: pid 19758 tid 19941 thread 19 bound to OS proc set 3
OMP: Info #250: KMP_AFFINITY: pid 19758 tid 19940 thread 18 bound to OS proc set 2
OMP: Info #250: KMP_AFFINITY: pid 19758 tid 19942 thread 20 bound to OS proc set 4
OMP: Info #250: KMP_AFFINITY: pid 19758 tid 19944 thread 22 bound to OS proc set 6
OMP: Info #250: KMP_AFFINITY: pid 19758 tid 19943 thread 21 bound to OS proc set 5
OMP: Info #250: KMP_AFFINITY: pid 19758 tid 19945 thread 23 bound to OS proc set 7
OMP: Info #250: KMP_AFFINITY: pid 19758 tid 19946 thread 24 bound to OS proc set 8
OMP: Info #250: KMP_AFFINITY: pid 19758 tid 19952 thread 25 bound to OS proc set 9
OMP: Info #250: KMP_AFFINITY: pid 19758 tid 19978 thread 26 bound to OS proc set 10
OMP: Info #250: KMP_AFFINITY: pid 19758 tid 19979 thread 27 bound to OS proc set 11
OMP: Info #250: KMP_AFFINITY: pid 19758 tid 19980 thread 28 bound to OS proc set 12
OMP: Info #250: KMP_AFFINITY: pid 19758 tid 19981 thread 29 bound to OS proc set 13
OMP: Info #250: KMP_AFFINITY: pid 19758 tid 19982 thread 30 bound to OS proc set 14
OMP: Info #250: KMP_AFFINITY: pid 19758 tid 19983 thread 31 bound to OS proc set 15
OMP: Info #250: KMP_AFFINITY: pid 19758 tid 19984 thread 32 bound to OS proc set 0
2100/2100 [==============================] - 1s 445us/step - loss: 10.5675 - acc: 0.3371 - val_loss: 11.0002 - val_acc: 0.3100
Epoch 2/5
2100/2100 [==============================] - 0s 16us/step - loss: 10.5675 - acc: 0.3371 - val_loss: 11.0002 - val_acc: 0.3100
Epoch 3/5
2100/2100 [==============================] - 0s 32us/step - loss: 10.5675 - acc: 0.3371 - val_loss: 11.0002 - val_acc: 0.3100
Epoch 4/5
2100/2100 [==============================] - 0s 15us/step - loss: 10.5675 - acc: 0.3371 - val_loss: 11.0002 - val_acc: 0.3100
Epoch 5/5
2100/2100 [==============================] - 0s 16us/step - loss: 10.5675 - acc: 0.3371 - val_loss: 11.0002 - val_acc: 0.3100

read chunk [1000:2000] from hdfs:///project/monitoring/archive/rucio/raw/events/2019/08/15/part-00000-ddb030f3-fd5d-4ecd-9497-b695919417d7-c000.json.gz label activity
....
```
