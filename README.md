### Machine Learning as a Service for HEP

[![Build Status](https://travis-ci.org/vkuznet/MLaaS4HEP.svg?branch=master)](https://travis-ci.org/vkuznet/MLaaS4HEP)
[![License:MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/vkuznet/LICENSE)
[![DOI](https://zenodo.org/badge/156857396.svg)](https://zenodo.org/badge/latestdoi/156857396)
[![Tweet](https://img.shields.io/twitter/url/http/shields.io.svg?style=social)](https://twitter.com/intent/tweet?text=Machine%20Learning%20as%20a%20service%20for%20HEP%20community&url=https://github.com/vkuznet/MLaaS4HEP&hashtags=python,ml)

MLaaS for HEP is a set of Python based modules to support reading HEP data and
stream them to ML of user choice for training. It consists of three independent layers:
- data streaming layer to handle remote data,
  see [reader.py](https://github.com/vkuznet/MLaaS4HEP/blob/master/src/python/reader.py)
- data training layer to train ML model for given HEP data,
  see [workflow.py](https://github.com/vkuznet/MLaaS4HEP/blob/master/src/python/workflow.py)
- data inference layer,
  see [tfaas_client.py](https://github.com/vkuznet/MLaaS4HEP/blob/master/src/python/tfaas_client.py)

The general architecture of MLaaS4HEP looks like this:
![MLaaS4HEP-architecture](https://github.com/vkuznet/MLaaS4HEP/blob/master/images/MLaaS4HEP_arch_gen.png)
Even though this architecture was originally developed for dealing with
HEP ROOT files we extend it to other data formats. So far the following
data formats are supported: JSON, CSV, Parquet, ROOT. The former ones support
reading files from local file system or HDFS, while later (ROOT) format allows
to read ROOT files from local file system or remote files via xrootd protocol.

The pre-trained models can be easily uploaded to
[TFaas](https://github.com/vkuznet/TFaaS) inference server for serving them to clients.

### Dependencies
The MLaaS4HEP relies on third-party libraries to support reading different
data-formats. Here we outline main of them:
- [pyarrow](https://arrow.apache.org) for reading data from HDFS file system
- [uproot](https://github.com/scikit-hep/uproot) for reading ROOT files
- [numpy](https://www.numpy.org), [pandas](https://pandas.pydata.org) for data representation
- [modin](https://github.com/modin-project/modin) for fast panda support
- [numba](https://numba.pydata.org) for speeing up individual functions
For ML modeling you may use your favorite framework, e.g. Keras, TensorFlow,
scikit-learn, PyTorch, etc.
Therefore, we suggest to use [anaconda](https://anaconda.org) to install its dependencies:
```
# to install pyarrow, uproot
conda install -c conda-forge pyarrow uproot numba scikit-learn
# to install pytorch
conda install -c pytorch pytorch
# to install TensorFlow, Kearas, Numpy, Pandas
conda install keras numpy pandas
```

### Instalation
The easiest way to install and run
[MLaaS4HEP](https://cloud.docker.com/u/veknet/repository/docker/veknet/mlaas4hep)
and
[TFaaS](https://cloud.docker.com/u/veknet/repository/docker/veknet/tfaas)
is to use pre-build docker images
```
# run MLaaS4HEP docker container
docker run veknet/mlaas4hep
# run TFaaS docker container
docker run veknet/tfaas
```

### Reading ROOT files
MLaaS4HEP python repository provides two base modules to read and manipulate with
HEP ROOT files. The `reader.py` module defines a DataReader class which is
able to read either local or remote ROOT files (via xrootd). And, `workflow.py`
module provide a basic DataGenerator class which can be used with any ML
framework to read HEP ROOT data in chunks. Both modules are based on
[uproot](https://github.com/scikit-hep/uproot) framework.

Basic usage
```
# setup the proper environment, e.g. 
# export PYTHONPATH=/path/src/python # path to MLaaS4HEP python framework
# export PATH=/path/bin:$PATH # path to MLaaS4HEP binaries

# get help and option description
reader --help

# here is a concrete example of reading local ROOT file:
reader --fin=/opt/cms/data/Tau_Run2017F-31Mar2018-v1_NANOAOD.root --info --verbose=1 --nevts=2000

# here is an example of reading remote ROOT file:
reader --fin=root://cms-xrd-global.cern.ch//store/data/Run2017F/Tau/NANOAOD/31Mar2018-v1/20000/6C6F7EAE-7880-E811-82C1-008CFA165F28.root --verbose=1 --nevts=2000 --info

# both of aforementioned commands produce the following output
First pass: 2000 events, 35.4363200665 sec, shape (2316,) 648 branches: flat 232 jagged
VMEM used: 960.479232 (MB) SWAP used: 0.0 (MB)
Number of events  : 1131872
# flat branches   : 648
...  # followed by a long list of ROOT branches found along with their dimentionality
TrigObj_pt values in [5.03515625, 1999.75] range, dim=21
```

More examples about using uproot may be found
[here](https://github.com/jpivarski/jupyter-talks/blob/master/2017-10-13-lpc-testdrive/uproot-introduction-evaluated.ipynb)
and
[here](https://github.com/jpivarski/jupyter-talks/blob/master/2017-10-13-lpc-testdrive/nested-structures-evaluated.ipynb)

### How to train ML model on HEP ROOT data
The HEP data are presented in [ROOT](https://root.cern.ch/) data-format.
The [DataReader](https://github.com/vkuznet/MLaaS4HEP/blob/master/src/python/reader.py#L188)
class provides access to ROOT files and various APIs to access the HEP data.

A simple workflow example can be found in
[workflow.py](https://github.com/vkuznet/MLaaS4HEP/blob/master/src/python/workflow.py)
code. It contains two examples, one for PyTorch and another for Keras.  It
contains two examples (on for PyTorch and another for TF in Keras) and show
full HEP ML workflow, i.e. it can read remote files and perform the training of
ML models with HEP ROOT files.


If you clone the repo and setup your PYTHONPATH you should be able to run it as
simple as

```
# setup the proper environment, e.g. 
# export PYTHONPATH=/path/src/python # path to MLaaS4HEP python framework
# export PATH=/path/bin:$PATH # path to MLaaS4HEP binaries

workflow --help

# run the code with list of LFNs from files.txt and using labels file labels.txt
workflow --files=files.txt --labels=labels.txt

# run pytorch example
workflow --files=files.txt --labels=labels.txt --model=ex_pytorch.py

# run keras example
workflow --files=files.txt --labels=labels.txt --model=ex_keras.py

# cat files.txt
#dasgoclient -query="file dataset=/Tau/Run2018C-14Sep2018_ver3-v1/NANOAOD"
/store/data/Run2018C/Tau/NANOAOD/14Sep2018_ver3-v1/60000/069A01AD-A9D0-7C4E-8940-FA5990EDFFCE.root
/store/data/Run2018C/Tau/NANOAOD/14Sep2018_ver3-v1/60000/577AF166-478C-1F40-8E10-044AA4BC0576.root
/store/data/Run2018C/Tau/NANOAOD/14Sep2018_ver3-v1/60000/9A661A77-58AC-0245-A442-8093D48A6551.root
/store/data/Run2018C/Tau/NANOAOD/14Sep2018_ver3-v1/60000/C226A004-077B-7E41-AFB3-6AFB38D1A63B.root
/store/data/Run2018C/Tau/NANOAOD/14Sep2018_ver3-v1/60000/D1E05C97-DB14-3941-86E8-C510D602C0B9.root
/store/data/Run2018C/Tau/NANOAOD/14Sep2018_ver3-v1/60000/6FA4CC7C-8982-DE4C-BEED-C90413312B35.root
/store/data/Run2018C/Tau/NANOAOD/14Sep2018_ver3-v1/60000/282E0083-6B41-1F42-B665-973DF8805DE3.root

# cat labels.txt
1
0
1
0
1
1
1

# run keras example and save our model into external file
workflow --files=files.txt --labels=labels.txt --model=ex_keras.py --fout=model.pb
```

The `workflow.py` relies on two JSON files, one which contains parameters for
reading ROOT files and another with specification of ROOT branches. The later
will be generated by reading ROOT file itself.

### How to train data using other data-formats
You may use `workflow.py` to use other data-formats, e.g. CSV, JSON, Parquet,
to train your model. The procedure is identical to dealing with HEP ROOT files.
```
# prepare your files.txt and labels.txt files, e.g. here we show example
# of using json gzipped files located on HDFS
cat files.txt
hdfs:///path/file1.json.gz
hdfs:///path/file2.json.gz

# optionally define your preprocessing function, see example in ex_preproc.py

# run workflow with your set of files, labels, model and preprocessing function
# and save it into model.pb file
workflow --files=files.txt --labels=labels.txt --model=ex_keras.py --preproc=ex_preproc.py --fout=model.pb
```

We provide more comprehensive example over
[here](doc/hdfs-example.md)

### HEP resnet
We provided full code called `hep_resnet.py` as a basic model based on
[ResNet](https://github.com/raghakot/keras-resnet) implementation.
It can classify images from HEP events, e.g.
```
hep_resnet.py --fdir=/path/hep_images --flabels=labels.csv --epochs=200 --mdir=models
```
Here we supply input directory `/path/hep_images` which contains HEP images
in `train` folder along with `labels.csv` file which provides labels.
The model runs for 200 epochs and save Keras/TF model into `models` output
directory.

### TFaaS inference server
We provide inference server in separate
[TFaaS](https://github.com/vkuznet/tfaas)
repository. It contains full set of instructions how to build and set it up.

### TFaaS client
To access your ML model in TFaaS inference server you only need to rely
on HTTP protocol. Please see [TFaaS](https://github.com/vkuznet/tfaas)
repository for more information.

But for convenience we also provide pure python
[client](https://github.com/vkuznet/TFaaS/blob/master/src/python/tfaas_client.py)
to perform all necessary actions against TFaaS server. Here is short
description of available APIs:

```
# setup url to point to your TFaaS server
url=http://localhost:8083

# create upload json file, which should include
# fully qualified model file name
# fully qualified labels file name
# model name you want to assign to your model file
# fully qualified parameters json file name
# For example, here is a sample of upload json file
{
    "model": "/path/model_0228.pb",
    "labels": "/path/labels.txt",
    "name": "model_name",
    "params":"/path/params.json"
}

# upload given model to the server
tfaas_client.py --url=$url --upload=upload.json

# list existing models in TFaaS server
tfaas_client.py --url=$url --models

# delete given model in TFaaS server
tfaas_client.py --url=$url --delete=model_name

# prepare input json file for querying model predictions
# here is an example of such file
{"keys":["attribute1", "attribute2"], values: [1.0, -2.0]}

# get predictions from TFaaS server
tfaas_client.py --url=$url --predict=input.json

# get image predictions from TFaaS server
# here we refer to uploaded on TFaaS ImageModel model
tfaas_client.py --url=$url --image=/path/file.png --model=ImageModel
```

### Citation
Please use this publication for further citation:
[http://arxiv.org/abs/1811.04492](http://arxiv.org/abs/1811.04492)
