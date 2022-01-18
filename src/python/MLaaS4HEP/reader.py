#!/usr/bin/env python
#-*- coding: utf-8 -*-
#pylint: disable=E1004,R0912,R0913,R0914,R0915,R0903,R0902,C0302
"""
File       : reader.py
Author     : Valentin Kuznetsov <vkuznet AT gmail dot com>
Reader module provides various readers for different data-formats:
JSON, CSV, Parquet, ROOT. It also allows access files either on local file
system, HDFS or viw xrootd (for ROOT data format). The support for HDFS
is provided by pyarrow library while for xrootd via uproot one.

There are 3 parameters each reader uses: nevts, chunk_size, nrows
nevts represents total number of events to read (for non-ROOT files it
is assigned to chunk_size). chunk_size is total buffer of events to read
from a file, nrows is total number events represented in a file.
RootDataReader reads data in chunks while others read entire file.
"""
from __future__ import print_function, division, absolute_import

# system modules
import os
import sys
import time
import json
import random
import argparse
import traceback
import itertools

import gzip

# numpy modules
import numpy as np

# pandas modules
# pd = None
# try: # https://github.com/modin-project/modin
#     import modin.pandas as pd
# except ImportError:
#     try:
#         import pandas as pd
#     except ImportError:
#         pass

# uproot
try:
    import uproot
except ImportError:
    pass

# numba
# try:
#     from numba import jit
# except ImportError:
#     def jit(f):
#         "Simple decorator which calls underlying function"
#         def new_f():
#             "Action function"
#             f()
#         return new_f

# psutil
try:
    import psutil
except ImportError:
    psutil = None

# histogrammar
try:
    import histogrammar as hg
#     import matplotlib
#     matplotlib.use('Agg')
#     from matplotlib.backends.backend_pdf import PdfPages
#     import matplotlib.pyplot as plt
except ImportError:
    hg = None

# pyarrow module for accessing HDFS
# https://wesmckinney.com/blog/python-hdfs-interfaces/
# https://arrow.apache.org
try:
    import pyarrow
    import pyarrow.parquet as pq
except ImportError:
    pyarrow = None

# MLaaS4HEP modules
from utils import nrows, dump_histograms, mem_usage, performance
from utils import steps, fopen, file_type, load_code

class OptionParser(object):
    "Option parser class for reader arguments"
    def __init__(self):
        "User based option parser"
        self.parser = argparse.ArgumentParser(prog='PROG')
        self.parser.add_argument("--fin", action="store", \
            dest="fin", default="", help="Input ROOT file")
        self.parser.add_argument("--fout", action="store", \
            dest="fout", default="", help="Output file name for ROOT specs")
        self.parser.add_argument("--preproc", action="store", \
            dest="preproc", default=None, \
            help="File name containing pre-processing function")
        self.parser.add_argument("--nan", action="store", \
            dest="nan", default=np.nan, \
            help="NaN value for padding, default np.nan [for ROOT file]")
        self.parser.add_argument("--branch", action="store", \
            dest="branch", default="Events", \
            help="Input ROOT file branch, default Events [for ROOT file]")
        self.parser.add_argument("--identifier", action="store", \
            dest="identifier", default="run,event,luminosityBlock", \
            help="Event identifier, default run,event,luminosityBlock [for ROOT file]")
        self.parser.add_argument("--branches", action="store", \
            dest="branches", default="", \
            help="Comma separated list of branches to read, default all [for ROOT file]")
        self.parser.add_argument("--exclude-branches", action="store", \
            dest="exclude_branches", default="", \
            help="Comma separated list of branches to exclude, default None [for ROOT file]")
        self.parser.add_argument("--nevts", action="store", \
            dest="nevts", default=5, \
            help="number of events to process, default 5, use -1 to read all events)")
        self.parser.add_argument("--chunk-size", action="store", \
            dest="chunk_size", default=1000, help="Chunk size (nevts) to read, default 1000")
        self.parser.add_argument("--specs", action="store", \
            dest="specs", default=None, \
            help="Input specs file [for ROOT file]")
        self.parser.add_argument("--redirector", action="store", \
            dest="redirector", default='root://cms-xrd-global.cern.ch', \
            help="XrootD redirector, default root://cms-xrd-global.cern.ch [for ROOT file]")
        self.parser.add_argument("--info", action="store_true", \
            dest="info", default=False, \
            help="Provide info about ROOT tree [for ROOT file]")
        self.parser.add_argument("--hists", action="store_true", \
            dest="hists", default=False, help="Create historgams for ROOT tree")
        self.parser.add_argument("--verbose", action="store", \
            dest="verbose", default=0, help="verbosity level")

def dim_jarr(arr):
    "Return dimention (max length) of jagged array"
    jdim = 0
    for item in arr:
        if jdim < len(item):
            jdim = len(item)
    return jdim

def min_max_arr(jagged_key, key, arr):
    """
    Helper function to find out min/max values of given array.
    The array can be either jagged one or normal numpy.ndarray
    """
    try:
        if key in jagged_key:
            arr = np.concatenate(arr, axis = 0)
        return float(np.min(arr)), float(np.max(arr))
    except ValueError:
        return 1e15, -1e15

class FileReader(object):
    """
    FileReader represents generic interface to read data files
    """
    def __init__(self, fin, label, chunk_size=1000, nevts=-1, preproc=None, verbose=0, reader=None):
        self.fin = fin
        self.label = label
        self.chunk_size = chunk_size if nevts == -1 else nevts
        self.nevts = nevts
        self.preproc = preproc
        self.verbose = verbose
        self.keys = []
        self.nrows = 0
        self.reader = reader
        self.istream = None
        self.type = self.__class__.__name__
        if not fin.lower().startswith('hdfs://'):
            if hasattr(fin, 'readline'): # we already given a file descriptor
                self.istream = fin
            else:
                self.istream = fopen(fin, 'r')
        if self.verbose:
            if self.reader:
                print('init {} with {}'.format(self.type, self.reader))
            else:
                print('init {}'.format(self.type))

    def info(self):
        "Provide basic info about class attributes"
        print('{} {}'.format(self.type, self))
        mkey = max([len(k) for k in self.__dict__])
        for key, val in self.__dict__.items():
            pad = ' ' * (mkey - len(key))
            print('{}{}: {}'.format(key, pad, val))

    def __exit__(self, gtype, value, gtraceback):
        "Exit function for our class"
        if self.istream and hasattr(self.istream, 'close'):
            self.istream.close()
        if self.reader:
            self.reader.__exit__()

    @property
    def columns(self):
        "Return names of columns of our data"
        if self.reader:
            return self.reader.columns
        return self.keys

    def next(self):
        "Read next chunk of data from out file"
        return self.reader.next() if self.reader else []

#
# HDFS readers
#

def hdfs_read(fin):
    "Read data from the fiven file into numpy array"
    if pyarrow:
        client = pyarrow.hdfs.connect()
        with client.open(fin) as istream:
            raw = istream.read()
            if fin.endswith('gz'):
                raw = getattr(gzip, "decompress")(raw)
            return raw
    else:
        raise Exception("pyarrow is not available")

class HDFSReader(FileReader):
    """
    HDFSReader represents interface to read data file from HDFS.
    """
    def __init__(self, fin, label, chunk_size=1000, nevts=-1, preproc=None, verbose=0):
        if  sys.version.startswith('3.'):
            super().__init__(fin, label, chunk_size, nevts, preproc, verbose)
        else:
            super(HDFSReader, self).__init__(\
                    fin, label, chunk_size, nevts, preproc, verbose)
        self.raw = None
        self.keys = None
        self.pos = 0

    def getdata(self):
        "Read next chunk of data from out file"
        if not self.raw:
            if self.verbose:
                print("%s reading %s" % (self.__class__.__name__, self.fin))
            time0 = time.time()
            self.raw = hdfs_read(self.fin)
            if self.verbose:
                print("read %s in %s sec" % (self.fin, time.time()-time0))
        return self.raw

class HDFSJSONReader(HDFSReader):
    """
    HDFSJSONReader represents interface to read JSON file from HDFS.
    """
    def __init__(self, fin, label, chunk_size=1000, nevts=-1, preproc=None, verbose=0):
        if  sys.version.startswith('3.'):
            super().__init__(fin, label, chunk_size, nevts, preproc, verbose)
        else:
            super(HDFSJSONReader, self).__init__(\
                    fin, label, chunk_size, nevts, preproc, verbose)

    def next(self):
        "Read next chunk of data from out file"
        lines = self.getdata().splitlines()
        if not self.nrows:
            self.nrows = len(lines)
        for idx in range(self.chunk_size):
            time0 = time.time()
            if len(lines) <= self.pos:
                break
            line = lines[self.pos]
            if not line:
                continue
            rec = json.loads(line.decode('utf-8'))
            if not rec:
                continue
            if self.preproc:
                rec = self.preproc(rec)
            if not self.keys:
                self.keys = [k for k in sorted(rec.keys())]
            if self.keys != sorted(rec.keys()):
                rkeys = sorted(rec.keys())
                msg = 'WARNING: record %s contains different set of keys from original ones\n' % idx
                msg += 'original keys : %s\n' % json.dumps(self.keys)
                msg += 'record   keys : %s\n' % json.dumps(rkeys)
                if len(self.keys) > len(rkeys):
                    diff = set(self.keys)-set(rkeys)
                    msg += 'orig-rkeys diff: %s\n' % diff
                else:
                    diff = set(rkeys)-set(self.keys)
                    msg += 'rkeys-orig diff: %s\n' % diff
                if self.verbose > 1:
                    print(msg)
            if self.label in self.keys:
                data = [rec.get(k, 0) for k in self.keys if k != self.label]
                label = rec[self.label]
            else:
                data = [rec.get(k, 0) for k in self.keys]
                label = self.label
            self.pos += 1
            data = np.array(data)
            if self.verbose > 1:
                print("read data chunk", self.pos, time.time()-time0, \
                        self.chunk_size, np.shape(data))
            yield data, label

class HDFSCSVReader(HDFSReader):
    """
    HDFSCSVReader represents interface to read CSV file from HDFS storage
    """
    def __init__(self, fin, label, chunk_size=1000, nevts=-1, preproc=None, \
            verbose=0, headers=None, separator=','):
        if  sys.version.startswith('3.'):
            super().__init__(fin, label, chunk_size, nevts, preproc, verbose)
        else:
            super(HDFSCSVReader, self).__init__(\
                    fin, label, chunk_size, nevts, preproc, verbose)
        self.headers = headers
        self.sep = separator

    def next(self):
        "Read next chunk of data from out file"
        lines = self.getdata().splitlines()
        if not self.nrows:
            self.nrows = len(lines)
        for idx in range(self.chunk_size):
            time0 = time.time()
            if len(lines) <= self.pos:
                break
            line = lines[self.pos]
            if not line:
                continue
            row = line.split(self.sep)
            if not row:
                continue
            if self.preproc:
                row = self.preproc(row)
            if not self.keys:
                self.keys = [k for k in sorted(row)]
                continue
            rec = dict(zip(self.keys, row))
            if self.keys != sorted(rec.keys()):
                msg = 'WARNING: record %s contains different set of keys from original ones' % idx
                if self.verbose:
                    print(msg)
            if self.label in self.keys:
                data = [rec.get(k, 0) for k in self.keys if k != self.label]
                label = rec[self.label]
            else:
                data = [rec.get(k, 0) for k in self.keys]
                label = self.label
            self.pos += 1
            data = np.array(data)
            if self.verbose > 1:
                print("read data chunk", self.pos, time.time()-time0, \
                        self.chunk_size, np.shape(data))
            yield data, label

class ParquetReader(HDFSReader):
    """
    ParquetReader represents interface to read Parque files
    """
    def __init__(self, fin, label, chunk_size=1000, nevts=-1, preproc=None, verbose=0):
        if  sys.version.startswith('3.'):
            super().__init__(fin, label, chunk_size, nevts, preproc, verbose)
        else:
            super(ParquetReader, self).__init__(\
                    fin, label, chunk_size, nevts, preproc, verbose)
        self.pos = 0

    def next(self):
        "Read next chunk of data from out file"
        data = pq.read_table(self.fin)
        xdf = data.to_pandas()
        shape = np.shape(xdf)
        self.nrows = shape[0]
        end = self.chunk_size if self.pos + self.chunk_size < shape[0] else shape[0]
        xdf = xdf[self.pos:end]
        self.pos = end
        self.keys = list(xdf.columns)
        if self.label in self.keys:
            label = xdf[self.label]
            xdf.drop(self.label, axis=1)
            yield xdf.values, label
        else:
            yield xdf.values, self.label

#
# Data reader classes
#

class JSONReader(FileReader):
    """
    JSONReader represents interface to read JSON file from local file system
    """
    def __init__(self, fin, label, chunk_size=1000, nevts=-1, preproc=None, verbose=0):
        if  sys.version.startswith('3.'):
            super().__init__(fin, label, chunk_size, nevts, preproc, verbose)
        else:
            super(JSONReader, self).__init__(\
                    fin, label, chunk_size, nevts, preproc, verbose)
        self.nrows = nrows(fin)

    def next(self):
        "Read next chunk of data from out file"
        for idx in range(self.chunk_size):
            line = self.istream.readline()
            if not line:
                continue
            rec = json.loads(line)
            if not rec:
                continue
            if self.preproc:
                rec = self.preproc(rec)
            if not self.keys:
                self.keys = [k for k in sorted(rec.keys())]
            if self.keys != sorted(rec.keys()):
                rkeys = sorted(rec.keys())
                msg = 'WARNING: record %s contains different set of keys from original ones\n' % idx
                msg += 'original keys : %s\n' % json.dumps(self.keys)
                msg += 'record   keys : %s\n' % json.dumps(rkeys)
                if len(self.keys) > len(rkeys):
                    diff = set(self.keys)-set(rkeys)
                    msg += 'orig-rkeys diff: %s\n' % diff
                else:
                    diff = set(rkeys)-set(self.keys)
                    msg += 'rkeys-orig diff: %s\n' % diff
                if self.verbose:
                    print(msg)
            if self.label in self.keys:
                data = [rec.get(k, 0) for k in self.keys if k != self.label]
                label = rec[self.label]
            else:
                data = [rec.get(k, 0) for k in self.keys]
                label = self.label
            yield np.array(data), label

class CSVReader(FileReader):
    """
    CSVReader represents interface to read CSV file from local file system
    """
    def __init__(self, fin, label, chunk_size=1000, nevts=-1, preproc=None, \
            verbose=0, headers=None, separator=','):
        if  sys.version.startswith('3.'):
            super().__init__(fin, label, chunk_size, nevts, preproc, verbose)
        else:
            super(CSVReader, self).__init__(\
                    fin, label, chunk_size, nevts, preproc, verbose)
        self.headers = headers
        self.keys = headers if headers else None
        self.sep = separator
        self.nrows = nrows(fin)

    def next(self):
        "Read next chunk of data from out file"
        for idx in range(self.chunk_size):
            line = self.istream.readline()
            if not line:
                continue
            row = line.split(self.sep)
            if not row:
                continue
            if self.preproc:
                row = self.preproc(row)
            if not self.keys:
                self.keys = [k for k in sorted(row)]
                continue
            rec = dict(zip(self.keys, row))
            if self.keys != sorted(rec.keys()):
                msg = 'WARNING: record %s contains different set of keys from original ones' % idx
                if self.verbose:
                    print(msg)
            if self.label in self.keys:
                data = [rec.get(k, 0) for k in self.keys if k != self.label]
                label = rec[self.label]
            else:
                data = [rec.get(k, 0) for k in self.keys]
                label = self.label
            self.nrows += 1
            yield np.array(data), label

class AvroReader(FileReader):
    """
    AvroReader represents interface to read Avro file.
    Depends on: https://issues.apache.org/jira/browse/ARROW-1209
    """
    def __init__(self, fin, label, chunk_size=1000, nevts=-1, preproc=None, verbose=0):
        if  sys.version.startswith('3.'):
            super().__init__(fin, label, chunk_size, nevts, preproc, verbose)
        else:
            super(AvroReader, self).__init__(\
                    fin, label, chunk_size, nevts, preproc, verbose)

    def next(self):
        "Read next chunk of data from out file"
        raise NotImplementedError

#
# User based classes
#
class JsonReader(FileReader):
    """
    JsonReader represents interface to read jSON file either from local file system or HDFS
    """
    def __init__(self, fin, label, chunk_size=1000, nevts=-1, preproc=None, verbose=0):
        if  fin.lower().startswith('hdfs://'):
            reader = HDFSJSONReader(fin, label, chunk_size, nevts, preproc)
        else:
            reader = JSONReader(fin, label, chunk_size, nevts, preproc)
        if  sys.version.startswith('3.'):
            super().__init__(fin, label, chunk_size, nevts, preproc, verbose, reader)
        else:
            super(JsonReader, self).__init__(\
                    fin, label, chunk_size, nevts, preproc, verbose, reader)
        self.nrows = reader.nrows


class CsvReader(FileReader):
    """
    CsvReader represents interface to read CSV file either from local file system or HDFS
    """
    def __init__(self, fin, label, chunk_size=1000, nevts=-1, preproc=None, verbose=0):
        if  fin.lower().startswith('hdfs://'):
            reader = HDFSCSVReader(fin, label, chunk_size, nevts, preproc)
        else:
            reader = CSVReader(fin, label, chunk_size, nevts, preproc)
        if  sys.version.startswith('3.'):
            super().__init__(fin, label, chunk_size, nevts, preproc, verbose, reader)
        else:
            super(CsvReader, self).__init__(\
                    fin, label, chunk_size, nevts, preproc, verbose, reader)
        self.nrows = reader.nrows


class RootDataReader(object):
    """
    RootDataReader class provide interface to read ROOT files
    and APIs to access its data. It uses two-pass procedure
    unless specs file is provided. The first pass parse entire
    file and identifies flat/jagged keys, their dimensionality
    and min/max values. All of them are stored in a file specs.
    The second pass uses specs to convert jagged structure of
    ROOT file into flat DataFrame format.
    """
    def __init__(self, fin, branch='Events', selected_branches=None, \
            exclude_branches=None, identifier=None, label=None, \
            chunk_size=1000, nevts=-1, specs=None, nan=np.nan, histograms=False, \
            redirector='root://cms-xrd-global.cern.ch', verbose=0):
        self.type = self.__class__.__name__
        self.fin = xfile(fin, redirector)
        self.verbose = verbose
        if self.verbose:
            print("Reading {}".format(self.fin))
        self.istream = uproot.open(self.fin)
        self.branches = {}
        self.gen = None
        self.out_branches = []
        self.identifier = identifier if identifier else ['run', 'event', 'luminosityBlock']
        self.tree = self.istream[branch]
        self.nrows = self.tree.num_entries
        self.nevts = nevts if nevts != -1 else self.nrows
        self.label = label
        self.idx = -1
        self.chunk_idx = 0
        self.chunk_size = chunk_size if chunk_size < self.nrows else self.nrows
        self.nan = float(nan)
        self.attrs = []
        self.shape = None
        self.cache = {}
        self.hdict = {}
        self.hists = histograms
        self.idx_label = 0
        self.flat_keys_encoded = []
        self.jagged_keys_encoded = []
        self.keys = []
        self.min_list = []
        self.max_list = []
        self.jdimension = []
        self.dimension_list = []
        self.time_reading = []
        self.time_reading_and_specs = []
        if specs:
            self.load_specs(specs)
        else:
            self.jdim = {}
            self.minv = {}
            self.maxv = {}
            self.jkeys = []
            self.fkeys = []
            self.nans = {}

        time0 = time.time()
        if exclude_branches:
            print(f"Excluded branches: {exclude_branches}")
            all_branches=self.tree.keys()
            exclude_branches=[elem for elem in exclude_branches]
            self.out_branches=[elem for elem in all_branches if (elem not in exclude_branches)]
        if selected_branches:
            print(f"Selected branches: {selected_branches}")
            selected_branches=[elem for elem in selected_branches]
            self.out_branches=[elem for elem in selected_branches]

        # perform initialization
        self.init()
        if self.verbose:
            print("{} init is complete in {} sec".format(self, time.time()-time0))

        # declare histograms for original and normilized values
        if hg and self.hists:
            for key in self.attrs:
                low = self.minv[key]
                high = self.maxv[key]
                self.hdict['%s_orig' % key] = \
                        hg.Bin(num=100, low=low, high=high, quantity=lambda x: x, value=hg.Count())
                self.hdict['%s_norm' % key] = \
                        hg.Bin(num=100, low=0, high=1, quantity=lambda x: x, value=hg.Count())


    def load_specs(self, specs):
        "load given specs"
        if not isinstance(specs, dict):
            if self.verbose:
                print(f"load specs from {specs} for {self.fin}")
            specs = json.load(open(specs))
        if self.verbose > 1:
            print("ROOT specs: {}".format(json.dumps(specs)))
        self.jdim = specs['jdim']
        self.minv = specs['minv']
        self.maxv = specs['maxv']
        self.jkeys = specs['jkeys']
        self.fkeys = specs['fkeys']
        self.nans = specs['nans']

        self.flat_keys_encoded = [key for key in self.flat_keys()]
        self.jagged_keys_encoded = [key for key in self.jagged_keys()]
        self.keys = self.flat_keys_encoded + self.jagged_keys_encoded
        self.min_list = [self.minv[key] for key in self.keys]
        self.max_list = [self.maxv[key] for key in self.keys]
        self.jdimension = [self.jdim[key] for key in self.jagged_keys_encoded]
        self.dimension_list = [1] * len(self.flat_keys_encoded)
        self.dimension_list = self.dimension_list + self.jdimension

    def fetch_data(self, key):
        "fetch data for given key from underlying ROOT tree"
        if key in self.branches:
            return self.branches[key]
        raise Exception('Unable to find "%s" key in ROOT branches' % key)

    def read_chunk(self, nevts, set_branches=False, set_min_max=False):
        "Reach chunk of events and determine min/max values as well as load branch values"
        # read some portion of the data to determine branches
        start_time = time.time()
        if not self.gen:
            if self.out_branches:
                self.gen = self.tree.iterate( \
                    self.out_branches, \
                    step_size=nevts, \
                    library="np")
            else:
                self.gen = self.tree.iterate( \
                    step_size=nevts, \
                    library='np')
        self.branches = {} # start with fresh dict
        try:
            self.branches = next(self.gen) # python 3.X and 2.X
        except StopIteration:
            if self.out_branches:
                self.gen = self.tree.iterate( \
                    branches=self.out_branches, \
                    step_size=nevts, \
                    library='np')
            else:
                self.gen = self.tree.iterate(step_size=nevts, library='np')
            self.branches = next(self.gen) # python 3.X and 2.X

        self.time_reading.append(time.time()-start_time)
        end_time = time.time()
        self.idx += nevts
        if self.verbose:
            performance(nevts, self.tree, self.branches, start_time, end_time)
        if set_branches:
            for key, val in self.branches.items():
                if isinstance(self.tree[key].interpretation, uproot.AsJagged):
                    self.jkeys.append(key)
                else:
                    self.fkeys.append(key)
                self.minv[key], self.maxv[key] = min_max_arr(self.jkeys, key, val)
        if set_min_max:
            for key, val in self.branches.items():
                minv, maxv = min_max_arr(self.jkeys, key, val)
                if minv < self.minv[key]:
                    self.minv[key] = minv
                if maxv > self.maxv[key]:
                    self.maxv[key] = maxv

    def columns(self):
        "Return columns of produced output vector"
        cols = self.flat_keys()
        for key in self.jagged_keys():
            for idx in range(self.jdim[key]):
                cols.append('%s_%s' % (key, idx))
        return cols

    def init(self):
        "Initialize class data members by scaning ROOT tree"
        if self.minv and self.maxv:
            self.attrs = sorted(self.flat_keys()) + sorted(self.jagged_keys())
            self.shape = len(self.flat_keys()) + sum(self.jdim.values())
            msg = "+++ first pass: %s events, (%s-flat, %s-jagged) branches, %s attrs" \
                    % (self.nrows, len(self.flat_keys()), len(self.jagged_keys()), self.shape)
            if self.verbose:
                print(msg)
            if self.verbose > 1:
                print("\n### Flat attributes:")
                for key in self.flat_keys():
                    print(key)
                print("\n### Jagged array attributes:")
                for key in self.jagged_keys():
                    print(key)
            self.idx = -1
            return

        if psutil and self.verbose:
            vmem0 = psutil.virtual_memory()
            swap0 = psutil.swap_memory()

        msg = ''

        # if self.nevnts=0 we'll use 2x self.chunk_size to determine
        # the dimensions otherwise job waits too long to possibly scan
        # all events in a file which can be too large.
        tot_rows = self.nrows
        if not self.nevts:
            tot_rows = 2*self.chunk_size
            if self.verbose:
                print("# will use {} events to obtain dimensionality".format(tot_rows))

        # scan all rows to find out largest jagged array dimension
        tot = 0
        set_branches = True
        set_min_max = True
        for chunk in steps(tot_rows, self.chunk_size):
            time_beginning = time.time()
            if tot + self.chunk_size > self.nevts:
                nevts = self.nevts - tot
                tot = self.nevts
            else:
                nevts = len(chunk) # chunk here contains event indexes
                tot += nevts
            self.read_chunk(nevts, set_branches=set_branches, set_min_max=set_min_max)
            set_branches = False # we do it once
            for key in self.jkeys:
                if key not in self.jdim:
                    self.jdim[key] = 0
                dim = dim_jarr(self.fetch_data(key))
                if dim > self.jdim.get(key, 0):
                    self.jdim[key] = dim
            self.time_reading_and_specs.append(time.time()-time_beginning)
            if self.nevts > 0 and tot >= self.nevts:
                if self.verbose:
                    print(f"###total time elapsed for reading + specs computing: {sum(self.time_reading_and_specs[:])}; number of chunks {len(self.time_reading_and_specs)}")
                    print(f"###total time elapsed for reading: {sum(self.time_reading[:])}; number of chunks {len(self.time_reading)}\n")
                    if self.nevts == self.nrows:
                        print(f"###total time elapsed for reading + specs computing: {sum(self.time_reading_and_specs[:-1])}; number of chunks {len(self.time_reading_and_specs)-1}")
                        print(f"###total time elapsed for reading: {sum(self.time_reading[:-1])}; number of chunks {len(self.time_reading)-1}\n")
                break

        # if we've been asked to read all or zero events we determine
        # number of events as all available rows in TTree which is set as
        # self.tree.numentries in __init__
        if self.nevts < 1:
            self.nevts = self.nrows

        # initialize all nan values (zeros) in normalize phase-space
        # this should be done after we get all min/max values
        for key in self.branches.keys():
            self.nans[key] = self.normalize(key, 0)

        # reset internal indexes since we done with first pass reading
        self.idx = -1
        self.gen = None

        # define final list of attributes
        self.attrs = sorted(self.flat_keys()) + sorted(self.jagged_keys())

        if self.verbose > 1:
            print("\n### Dimensionality")
            for key, val in self.jdim.items():
                print(key, val)
            print("\n### min/max values")
            for key, val in self.minv.items():
                print(key, val, self.maxv[key])
        self.shape = len(self.flat_keys()) + sum(self.jdim.values())
        msg = "--- first pass: %s events, (%s-flat, %s-jagged) branches, %s attrs" \
                % (self.nrows, len(self.flat_keys()), len(self.jagged_keys()), self.shape)
        if self.verbose:
            print(msg)
            if self.verbose > 1:
                print("\n### Flat attributes:")
                for key in self.flat_keys():
                    print(key)
                print("\n### Jagged array attributes:")
                for key in self.jagged_keys():
                    print(key)
        if psutil and self.verbose:
            vmem1 = psutil.virtual_memory()
            swap1 = psutil.swap_memory()
            mem_usage(vmem0, swap0, vmem1, swap1)

    def write_specs(self, fout):
        "Write specs about underlying file"
        out = {'jdim': self.jdim, 'minv': self.minv, 'maxv': self.maxv}
        out['fkeys'] = self.flat_keys()
        out['jkeys'] = self.jagged_keys()
        out['nans'] = self.nans
        if self.verbose:
            print("write {}".format(fout))
        with open(fout, 'w') as ostream:
            ostream.write(json.dumps(out))

    def next(self):
        "Provides read interface for next event using vectorize approach"
        self.idx = self.idx + 1
        # read new chunk of records if necessary
        if not self.idx % self.chunk_size:
            if self.idx + self.chunk_size > self.nrows:
                nevts = self.nrows - self.idx
            else:
                nevts = self.chunk_size
            self.read_chunk(nevts)
            self.chunk_idx = 0 # reset chunk index after we read the chunk of data
            self.idx = self.idx - nevts # reset index after chunk read by nevents offset
            if self.verbose > 1:
                print("idx", self.idx, "read", nevts, "events")

        # form DataFrame record
        try:
            rec = [self.branches[key][self.chunk_idx] for key in self.keys]
        except:
            if len(rec) <= self.chunk_idx:
                raise Exception("For key='%s' unable to find data at pos=%s while got %s" \
                    % (key, self.chunk_idx, len(self.branches[key])))
            print("failed key", key)
            print("failed idx", self.chunk_idx)
            print("len(fdata)", len(self.branches[key]))
            raise

        # normalise and adjust dimension of the events
        result = [x1 if x3 == x2 else (x1 - x2) / (x3 - x2) for (x1, x2, x3) in \
            zip(rec, self.min_list, self.max_list) ]
        result = [[result[i]] if i < len(self.flat_keys_encoded) else result[i].tolist() \
            if len(result[i]) == self.dimension_list[i] else \
            self.add_dim(result[i], i) for i in range(0, len(result))]
        xdf = list(itertools.chain.from_iterable(result))
        mask = list(np.isnan(xdf) * 1)
        self.chunk_idx = self.chunk_idx + 1
        return np.array(xdf), np.array(mask), self.idx_label

    def add_dim(self, elem, index):
        "Allows to extend dimension of an array after reading the max dimension from the specs file"
        a = np.empty(self.dimension_list[index]) * np.nan
        a[:elem.shape[0]] = elem
        return a.tolist()

    def next_old(self):
        '''Provides read interface for next event using vectorize approach
           This is the old function, slower than the new one. It is kept for completeness'''
        self.idx = self.idx + 1
        # build output matrix
        time0 = time.time()
        shape = len(self.flat_keys())
        for key in sorted(self.jagged_keys()):
            shape += self.jdim[key]
        xdf = np.ndarray(shape=(shape,))
        mask = np.ndarray(shape=(shape,), dtype=np.int)
        idx_label = 0

        # read new chunk of records if necessary
        if not self.idx % self.chunk_size:
            if self.idx + self.chunk_size > self.nrows:
                nevts = self.nrows - self.idx
            else:
                nevts = self.chunk_size
            self.read_chunk(nevts)
            self.chunk_idx = 0 # reset chunk index after we read the chunk of data
            self.idx = self.idx - nevts # reset index after chunk read by nevents offset
            if self.verbose > 1:
                print("idx", self.idx, "read", nevts, "events")

        # read event info
        event = []
        for key in self.identifier:
            fdata = self.fetch_data(key)
            if len(fdata) <= self.chunk_idx:
                raise Exception("For key='%s' unable to find data at pos=%s while got %s" \
                        % (key, self.chunk_idx, len(fdata)))
            event.append(fdata[self.chunk_idx])

        # form DataFrame record
        rec = {}
        for key in self.branches.keys():
            try:
                fdata = self.fetch_data(key)
                if len(fdata) <= self.chunk_idx:
                    raise Exception("For key='%s' unable to find data at pos=%s while got %s" \
                            % (key, self.chunk_idx, len(fdata)))
                rec[key] = fdata[self.chunk_idx]
            except:
                print("failed key", key)
                print("failed idx", self.chunk_idx)
                print("len(fdata)", len(fdata))
                raise

        # advance chunk index since we read the record
        self.chunk_idx = self.chunk_idx + 1

        idx = 0
        for idx, key in enumerate(sorted(self.flat_keys())):
            if sys.version.startswith('3.') and isinstance(key, str):
                key = key.encode('ascii') # convert string to binary
            if key.decode() != self.label:
                xdf[idx] = self.normalize(key, rec[key])
            else:
                idx_label = idx
                xdf[idx] = rec[key]
            if hg and self.hists:
                self.hdict['%s_orig' % key].fill(rec[key])
                if xdf[idx] != self.nan:
                    self.hdict['%s_norm' % key].fill(xdf[idx])
            mask[idx] = 1
        if idx: # only advance position if we read something from flat_keys
            pos = idx + 1 # position in xdf for jagged branches
        else:
            pos = 0

        for key in sorted(self.jagged_keys()):
            # check if key in our record
            if key in rec.keys():
                vals = rec.get(key, [])
            else: # if not convert key to bytes key and use it to look-up a value
                vals = rec.get(key.encode('utf-8'), [])
            for jdx in range(self.jdim[key]):
                # assign np.nan in case if we get empty array
                val = vals[jdx] if len(vals) > jdx else np.nan
                idx = pos+jdx
                xdf[idx] = self.normalize(key, val)
                if hg and self.hists:
                    self.hdict['%s_orig' % key].fill(val)
                    if xdf[idx] != self.nan:
                        self.hdict['%s_norm' % key].fill(xdf[idx])
                if np.isnan(val):
                    mask[idx] = 0
                else:
                    mask[idx] = 1
            pos = idx + 1

        if self.verbose > 1:
            print("# idx=%s event=%s shape=%s proc.time=%s" % (
                self.idx, event, np.shape(xdf), (time.time()-time0)))
            if self.idx < 3:
                # pick-up 3 branches for cross checking
                if self.jagged_keys():
                    aidx = [random.randint(0, len(self.jagged_keys())-1) for _ in range(3)]
                    try:
                        keys = [self.jagged_keys()[i] for i in aidx]
                        for key in keys:
                            data = self.tree[key].array()
                            idx = self.attrs.index(key)
                            start_idx, end_idx = self.find_branch_idx(key)
                            print("+ branch=%s, row %s, position %s:%s, min=%s max=%s" \
                                    % (key, self.idx, start_idx, end_idx, \
                                    self.minv[key], self.maxv[key]))
                            print("+ xdf", xdf[start_idx:end_idx])
                            print(data)
                    except:
                        print("aidx=%s, len(jagged_keys)=%s" % (aidx, len(self.jagged_keys())))
                        traceback.print_exc()
        return xdf, mask, idx_label

    def find_branch_idx(self, attr):
        "Find start and end indexes of given attribute"
        idx = self.attrs.index(attr)
        if attr in self.flat_keys():
            return idx, idx+1
        start_idx = len(self.flat_keys())
        for key in sorted(self.jagged_keys()):
            if key == attr:
                return start_idx, start_idx + self.jdim[key]
            start_idx += self.jdim[key]
        raise Exception("Unable to find branch idx for %s" % attr)

    def jagged_keys(self):
        "helper function to return list of jagged branches"
        jkeys = sorted(list(self.jkeys))
        return jkeys

    def flat_keys(self):
        "helper function to return list of normal branches"
        fkeys = [k for k in self.fkeys if k not in self.identifier]
        return sorted(fkeys)

    def draw_value(self, key):
        "Draw a random value from underlying chunk for a given key"
        data = self.branches[key] # jagged array
        # get random index for accessing element of jagged array
        while True:
            idx = random.randint(0, len(data)-1)
            values = data[idx]
            if values:
                if len(values) == 1:
                    val = values[0]
                else:
                    jdx = random.randint(0, len(values)-1)
                    val = values[jdx]
                if random.randint(0, 1):
                    return val + val/10.
                return val - val/10.

    def normalize(self, key, val):
        "Normalize given value to 0-1 range according to min/max values"
        if isinstance(key, bytes):
            key = key.decode()
        # in case if our value is np.nan we'll assign nan value given to class
        if np.isnan(val):
            return self.nan
        minv = float(self.minv.get(key, 0))
        maxv = float(self.maxv.get(key, 1))
        if maxv == minv:
            return val
        return (val-minv)/(maxv-minv)

    def denormalize(self, key, val):
        "De-normalize given value to 0-1 range according to min/max values"
        if val == 0:
            return self.nan
        minv = float(self.minv.get(key, 0))
        maxv = float(self.maxv.get(key, 1))
        return val*(maxv-minv)+minv

    def info(self):
        "Provide human readable form of ROOT branches"
        print("Number of events  : %s" % self.nrows)
        print("# flat branches   : %s" % len(self.flat_keys()))
        if self.verbose:
            for key in self.flat_keys():
                print("%s values in [%s, %s] range, dim=%s" % (
                    key,
                    self.minv.get(key, 'N/A'),
                    self.maxv.get(key, 'N/A'),
                    self.jdim.get(key, 'N/A')))
        print("# jagged branches : %s" % len(self.jagged_keys()))
        if self.verbose:
            for key in self.jagged_keys():
                print("%s values in [%s, %s] range, dim=%s" % (
                    key,
                    self.minv.get(key, 'N/A'),
                    self.maxv.get(key, 'N/A'),
                    self.jdim.get(key, 'N/A')))

def object_size(data):
    "Return size of the data"
    return sys.getsizeof(data.tobytes())

def size_format(uinput):
    """
    Format file size utility, it converts file size into KB, MB, GB, TB, PB units
    """
    try:
        num = float(uinput)
    except ValueError:
        traceback.print_exc()
        return "N/A"
    base = 1000. # CMS convention to use power of 10
    if  base == 1000.: # power of 10
        xlist = ['', 'KB', 'MB', 'GB', 'TB', 'PB']
    elif base == 1024.: # power of 2
        xlist = ['', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB']
    for xxx in xlist:
        if  num < base:
            return "%3.1f%s" % (num, xxx)
        num /= base
    return "N/A"

def parse(reader, nevts, fout, hists):
    "Parse given number of events from given reader"
    time0 = time.time()
    count = 0
    if nevts == -1:
        nevts = reader.nrows
    farr = []
    jarr = []
    if reader.type == 'RootDataReader':
        for _ in range(nevts):
            xdf, _mask = reader.next()
            fdx = len(reader.flat_keys())
            flat = xdf[:fdx]
            jagged = xdf[fdx:]
            fsize = object_size(flat)
            jsize = object_size(jagged)
            farr.append(fsize)
            jarr.append(jsize)
            count += 1
    else:
        for data, _ in reader.next():
            if not count:
                print(json.dumps(reader.columns))
            print(json.dumps(data.tolist()))
            count += 1
    if reader.type == 'root':
        print("avg(flat)=%s, avg(jagged)=%s, ratio=%s" \
                % (size_format(np.mean(farr)), \
                size_format(np.mean(jarr)), np.mean(farr)/np.mean(jarr)))
    tot_time = time.time()-time0
    print("Read %s evts, %s Hz, total time %s" % (
        count, count/tot_time, tot_time))
    if fout:
        reader.write_specs(fout)
    if hg and hists:
        hgkeys = [k for k in reader.attrs]
        dump_histograms(reader.hdict, hgkeys)

def write(reader, nevts, fout):
    "Write given number of events from given reader to NumPy"
    time0 = time.time()
    count = 0
    with open(fout, 'wb+') as ostream:
        if nevts == -1:
            nevts = reader.nrows
        for _ in range(nevts):
            xdf = reader.next()
            ostream.write(xdf.tobytes())
            count += 1
        tot_time = time.time()-time0
        print("Read %s evts, %s Hz, total time %s" % (
            count, count/tot_time, tot_time))

def xfile(fin, redirector='root://cms-xrd-global.cern.ch'):
    "Test if file is local or remote and setup proper prefix"
    if fin.startswith(redirector):
        return fin
    if os.path.exists(fin):
        return fin
    return "%s/%s" % (redirector, fin)

def main():
    "Main function"
    optmgr = OptionParser()
    opts = optmgr.parser.parse_args()
    fin = opts.fin
    fout = opts.fout
    verbose = int(opts.verbose)
    nevts = int(opts.nevts)
    chunk_size = int(opts.chunk_size)
    nan = float(opts.nan)
    nevts = int(opts.nevts)
    preproc = None
    if opts.preproc:
        preproc = load_code(opts.preproc, 'preprocessing')
    specs = opts.specs
    branch = opts.branch
    branches = opts.branches.split(',') if opts.branches else []
    exclude_branches = []
    if opts.exclude_branches:
        if os.path.isfile(opts.exclude_branches):
            exclude_branches = \
                    [r.replace('\n', '') for r in open(opts.exclude_branches).readlines()]
        else:
            exclude_branches = opts.exclude_branches.split(',')
    hists = opts.hists
    identifier = [k.strip() for k in opts.identifier.split(',')]
    label = None
    if file_type(fin) == 'root':
        reader = RootDataReader(fin, branch=branch, selected_branches=branches, \
                identifier=identifier, exclude_branches=exclude_branches, \
                histograms=hists, nan=nan, chunk_size=chunk_size, \
                nevts=nevts, specs=specs, redirector=opts.redirector, verbose=verbose)
    elif file_type(fin) == 'csv':
        reader = CsvReader(fin, label, chunk_size, nevts, preproc, verbose)
    elif file_type(fin) == 'json':
        reader = JsonReader(fin, label, chunk_size, nevts, preproc, verbose)
    elif file_type(fin) == 'parquet':
        reader = ParquetReader(fin, label, chunk_size, nevts, preproc, verbose)
    elif file_type(fin) == 'avro':
        reader = AvroReader(fin, label, chunk_size, nevts, preproc, verbose)
    if opts.info:
        reader.info()
    else:
        parse(reader, nevts, fout, hists)

if __name__ == '__main__':
    main()
