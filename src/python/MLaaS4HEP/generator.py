#!/usr/bin/env python
#-*- coding: utf-8 -*-
#pylint: disable=R0915,R0912,R0913,R0914,R0902
"""
File       : generator.py
Author     : Valentin Kuznetsov <vkuznet AT gmail dot com>
Generator module defines various data generator for MLaaS4HEP framework.
"""
from __future__ import print_function, division, absolute_import

# system modules
import os
import json
import time
import random

# numpy modules
import numpy as np

# MLaaS4HEP modules
from reader import RootDataReader, JsonReader, CsvReader, AvroReader, ParquetReader
from utils import file_type, timestamp, global_cut, print_cut


class MetaDataGenerator(object):
    """
    MetaDataGenerator class provides interface to read files.
    """
    def __init__(self, fin, labels, params=None, preproc=None, dtype=None):
        "Initialization function for Data Generator"
        time0 = time.time()
        self.dtype = str(dtype).lower()
        self.preproc = preproc
        if not params:
            params = {}
        # parse given parameters
        batch_size = params.get('batch_size', 256)
        self.verbose = params.get('verbose', 0)
        chunk_size = params.get('chunk_size', 1000)
        self.evts = params.get('nevts', -1)
        self.shuffle = params.get('shuffle', False)

        # convert input fin parameter into file list if necessary
        if isinstance(fin, str):
            self.files = [fin]
        elif isinstance(fin, list):
            self.files = fin
        else:
            raise Exception("Unsupported data-type '%s' for fin parameter" % type(fin))
        if isinstance(labels, str):
            self.labels = [labels for _ in range(len(self.files))]
        elif isinstance(labels, list):
            self.labels = labels
        else:
            raise Exception("Unsupported data-type '%s' for labels parameter" % type(labels))
        self.file_label_dict = dict(zip(self.files, self.labels))

        self.reader = {} # global reader will handle all files readers
        self.reader_counter = {} # reader counter keeps track of nevts read by readers

        if self.verbose:
            print(timestamp('Generator: {}'.format(self)))
            print("model parameters: {}".format(json.dumps(params)))

        self.start_idx = 0
        self.chunk_size = chunk_size
        self.stop_idx = chunk_size
        self.batch_size = batch_size

        # loop over files and create individual readers for them, then put them in a global reader
        for fname, label in self.file_label_dict.items():
            if self.dtype == 'json' or file_type(fname) == 'json':
                reader = JsonReader(fname, label, chunk_size=chunk_size, nevts=self.evts, \
                        preproc=self.preproc, verbose=self.verbose)
            elif self.dtype == 'csv' or file_type(fname) == 'csv':
                reader = CsvReader(fname, label, chunk_size=chunk_size, nevts=self.evts, \
                        preproc=self.preproc, verbose=self.verbose)
            elif self.dtype == 'avro' or file_type(fname) == 'avro':
                reader = AvroReader(fname, label, chunk_size=chunk_size, nevts=self.evts, \
                        preproc=self.preproc, verbose=self.verbose)
            elif self.dtype == 'parquet' or file_type(fname) == 'parquet':
                reader = ParquetReader(fname, label, chunk_size=chunk_size, nevts=self.evts, \
                        preproc=self.preproc, verbose=self.verbose)
            self.reader[fname] = reader
            self.reader_counter[fname] = 0

        self.current_file = self.files[0]

        print("init MetaDataGenerator in {} sec".format(time.time()-time0))
        print("available readers")
        for fname, reader in self.reader.items():
            print("{} {}".format(fname, reader))

    @property
    def nevts(self):
        "Return number of events of current file"
        return self.evts if self.evts != -1 else self.reader[self.current_file].nrows

    def __len__(self):
        "Return total number of batches this generator can deliver"
        return int(np.floor(self.nevts / self.batch_size))

    def next(self):
        "Return next batch of events"
        msg = "\nread chunk [{}:{}] from {} label {}"\
                .format(self.start_idx, self.stop_idx, self.current_file, \
                self.file_label_dict[self.current_file])
        gen = self.read_data(self.start_idx, self.stop_idx)
        # advance start and stop indecies
        self.start_idx = self.stop_idx
        self.stop_idx = self.start_idx+self.chunk_size
        if self.nevts != -1 and \
           (self.start_idx > self.nevts or \
           (self.reader[self.current_file].nrows and \
           self.start_idx > self.reader[self.current_file].nrows)):
            # we reached the limit of the reader
            self.start_idx = 0
            self.stop_idx = self.chunk_size
            raise StopIteration
        if self.verbose:
            print(msg)
        data = []
        labels = []
        for xdf, ldf in gen:
            data.append(xdf)
            labels.append(ldf)
        if not data:
            raise StopIteration
        data = np.array(data)
        labels = np.array(labels)
        if self.verbose:
            print("return shapes: data=%s labels=%s" % (np.shape(data), np.shape(labels)))
        return data, labels

    def __iter__(self):
        "Provide iterator capabilities to the class"
        return self

    def __next__(self):
        "Provide generator capabilities to the class"
        return self.next()

    def read_data(self, start=0, stop=100):
        "Helper function to read data via reader"
        # if we exceed number of events in a file we discard it
        if self.nevts < self.reader_counter[self.current_file]:
            if self.verbose:
                msg = "# discard {} since we read {} out of {} events"\
                        .format(self.current_file, \
                        self.reader_counter[self.current_file], self.nevts)
                print(msg)
            self.files.remove(self.current_file)
            if self.files:
                self.current_file = self.files[0]
            else:
                print("# no more files to read from")
                raise StopIteration
        if self.shuffle:
            idx = random.randint(0, len(self.files)-1)
            self.current_file = self.files[idx]
        current_file = self.current_file
        reader = self.reader[current_file]
        for data in reader.next():
            yield data
        if stop == -1:
            read_evts = reader.nrows
        else:
            read_evts = stop-start
        # update how many events we read from current file
        self.reader_counter[self.current_file] += read_evts
        if self.verbose:
            nevts = self.reader_counter[self.current_file]
            msg = "\ntotal read {} evts from {}".format(nevts, current_file)
            #print(msg)

class RootDataGenerator(object):
    """
    RootDataGenerator class provides interface to read HEP ROOT files.
    """
    def __init__(self, fin, labels, params=None, preproc=None, specs=None):
        "Initialization function for Data Generator"
        time0 = time.time()
        if not params:
            params = {}
        if preproc:
            self.preproc = preproc
        else:
            self.preproc = None
        # parse given parameters
        nan = params.get('nan', np.nan)
        batch_size = params.get('batch_size', 256)
        verbose = params.get('verbose', 0)
        branch = params.get('branch', 'Events')
        identifier = params.get('identifier', [])
        branches = params.get('selected_branches', [])
        chunk_size = params.get('chunk_size', 1000)
        exclude_branches = params.get('exclude_branches', [])
        redirector = params.get('redirector', 'root://cms-xrd-global.cern.ch')
        self.evts = params.get('nevts', -1)
        self.shuffle = params.get('shuffle', False)

        # convert input fin parameter into file list if necessary
        if isinstance(fin, str):
            self.files = [fin]
        elif isinstance(fin, list):
            self.files = fin
        else:
            raise Exception("Unsupported data-type '%s' for fin parameter" % type(fin))
        if isinstance(labels, str):
            self.labels = labels
        elif isinstance(labels, list):
            self.labels = labels
            self.file_label_dict = dict(zip(self.files, self.labels))
        else:
            raise Exception("Unsupported data-type '%s' for labels parameter" % type(labels))

        self.reader = {} # global reader will handle all files readers
        self.reader_counter = {} # reader counter keeps track of nevts read by readers

        if verbose:
            #print(timestamp('DataGenerator: {}'.format(self)))
            print("\nParameters: {}\n".format(json.dumps(params)))
        if self.preproc:
            print_cut(self.preproc)
        if exclude_branches and not isinstance(exclude_branches, list):
            if os.path.isfile(exclude_branches):
                exclude_branches = \
                        [r.replace('\n', '') for r in open(exclude_branches).readlines()]
            else:
                exclude_branches = exclude_branches.split(',')
            if verbose:
                print("exclude branches", exclude_branches)

        self.start_idx = 0
        self.chunk_size = chunk_size
        self.stop_idx = chunk_size
        self.batch_size = batch_size
        self.verbose = verbose
        self.jdim = {}
        self.minv = {}
        self.maxv = {}
        self.fkeys = []
        self.jkeys = []
        self.nans = {}
        self.gname = "global-specs.json"
        self.finish_label = False
        self.finish_file = False
        self.events = {'total': 0}
        self.evts_toread = {}

        # loop over files and create individual readers for them, then put them in a global reader
        for fname in self.files:
            # if no specs is given try to read them from local area
            fbase = fname.split('/')[-1].replace('.root', '')
            sname = 'specs-{}.json'.format(fbase)
            if not specs:
                if os.path.isfile(self.gname):
                    #if verbose:
                    #    print("\nloading specs {}\n".format(self.gname))
                    specs = json.load(open(self.gname))

            reader = RootDataReader(fname, branch=branch, identifier=identifier, label=self.labels,\
                    selected_branches=branches, exclude_branches=exclude_branches, \
                    nan=nan, chunk_size=chunk_size, nevts=self.evts, specs=specs, \
                    redirector=redirector, preproc=self.preproc, verbose=verbose)

            # build specs for the whole set of root files
            self.global_specs(fname, reader)


            if not os.path.isfile(sname):
                #if verbose:
                    #print("writing specs {}".format(sname))
                reader.write_specs(sname)

            self.reader[fname] = reader
            self.reader_counter[fname] = 0
        print('\n')
        for fname in self.files:
            self.reader[fname].load_specs(self.gname)
            if self.evts != -1:
                self.events[fname] = round((float(self.events[fname])/self.events['total'])*self.evts)
                if self.events[fname] == 0:
                    self.events[fname] = 1
                self.evts_toread[fname] = round((float(self.events[fname])/self.evts) * self.chunk_size)
            else:
                self.evts_toread[fname] = round((float(self.events[fname])/self.events['total']) * self.chunk_size)
        self.current_file = self.files[0]
        print("init RootDataGenerator in {} sec\n".format(time.time()-time0))


    @property
    def nevts(self):
        "Return number of events of current file"
        return self.evts if self.evts != -1 else self.reader[self.current_file].nrows

    def __len__(self):
        "Return total number of batches this generator can deliver"
        return int(np.floor(self.nevts / self.batch_size))

    def next(self):
        "Return next batch of events in form of data and mask vectors"
        data = []
        mask = []
        index_label = 0
        if self.shuffle:
            idx = random.randint(0, len(self.files)-1)
            self.current_file = self.files[idx]
        while self.check_file():
            pass
        gen = self.read_data(self.start_idx, self.stop_idx)
        for (xdf, mdf, idx_label) in gen:
            data.append(xdf)
            mask.append(mdf)
            index_label = idx_label
        if isinstance(self.labels, list):
            label = self.file_label_dict[self.current_file]
            data = np.array(data)
            mask = np.array(mask)
        else:
            # one branch contains the label
            if data:
                label = []
                c = list(zip(data,mask))
                random.shuffle(c)
                data, mask = zip(*c)
                label.append(np.array(data)[:,index_label])
                data = np.delete(np.array(data),index_label,1)
                mask = np.delete(np.array(mask),index_label,1)
        labels = np.full(shape=len(data), fill_value=label, dtype=np.int)
        return data, mask, labels

    def next_mix_files(self):
        '''Return next batch of events in form of data and mask vectors.
           Use it to equally mix events from different files'''
        if self.finish_file == True:
            raise StopIteration
        time_start = time.time()
        data = []
        mask = []
        for fname in self.files:
            if fname == self.files[0]:
                start = self.start_idx
            self.current_file = fname
            evts = self.evts_toread[fname]
            if evts + self.reader_counter[fname] >= self.events[fname]:
                self.finish_file = True
            if self.finish_file == False:
                if evts == 0: continue
                if fname == self.files[-1] and (self.start_idx + evts) % self.chunk_size != 0:
                    self.stop_idx = start + self.chunk_size
                    evts = self.stop_idx - self.start_idx
                else:
                    self.stop_idx = self.start_idx + evts
            else:
                evts = self.events[fname] - self.reader_counter[fname]
                self.stop_idx = self.start_idx + evts
            print(f"\nlabel {self.file_label_dict[self.current_file]}, "
            f"file <{self.current_file.split('/')[-1]}>, going to read {evts} events")
            gen = self.read_data_mix_files(self.start_idx, self.stop_idx)
            for (xdf, mdf, idx_label) in gen:
                data.append(xdf)
                mask.append(mdf)
            label = self.file_label_dict[self.current_file]
            if fname == self.files[0]:
                labels = np.full(shape=evts, fill_value=label, dtype=np.int)
            else:
                labels = np.append(labels, np.full(shape=evts, fill_value=label, dtype=np.int))
        data = np.array(data)
        mask = np.array(mask)
        if self.verbose:
            print(f"\nTime for handling a chunk: {time.time()-time_start}\n\n")
        return data, mask, labels

    def read_data_mix_files(self, start=0, stop=100):
        "Helper function to read ROOT data via uproot reader"
        msg = "read chunk [{}:{}] from {}"\
                .format(start, stop-1, self.current_file)
        if self.verbose:
            print(msg)
        current_file = self.current_file
        reader = self.reader[current_file]
        for _ in range(start, stop):
            xdf, mask, idx_label = reader.next()
            yield (xdf, mask, idx_label)
        read_evts = stop - start
        # update how many events we read from current file
        self.reader_counter[self.current_file] += read_evts
        # advance start and stop indecies
        self.start_idx = stop
        if self.verbose:
            nevts = self.reader_counter[self.current_file]
            msg = "total read {} evts from {}\n".format(nevts, current_file)
            #print(msg)


    def choose_file(self):
        if self.finish_label == False:
            for key, value in self.file_label_dict.items():
                if value != self.label_files:
                    self.current_file = key
                    break
                else:
                    if key == list(self.file_label_dict.keys())[-1]:
                        self.finish_label = True
                        idx = random.randint(0, len(self.files)-1)
                        self.current_file = self.files[idx]
        else:
            idx = random.randint(0, len(self.files)-1)
            self.current_file = self.files[idx]

    def next_mix_classes(self):
        '''Return next batch of events in form of data and mask vectors.
           Use it to equally mix events with different labels'''
        data = []
        mask = []

        #Read first file
        if self.shuffle:
            idx = random.randint(0, len(self.files)-1)
            self.current_file = self.files[idx]
        while self.check_file():
            pass
        gen = self.read_data(self.start_idx, self.stop_idx)
        for (xdf, mdf, idx_label) in gen:
            data.append(xdf)
            mask.append(mdf)
        label = self.file_label_dict[self.current_file]
        labels = np.full(shape=len(data), fill_value=label, dtype=np.int)
        print(f"label {self.file_label_dict[self.current_file]}, "
        f"file <{self.current_file.split('/')[-1]}>, read {len(labels)} events")
        self.label_files = label

        #Read second file
        self.choose_file()
        while self.check_file():
            pass
        self.label_files = self.file_label_dict[self.current_file]
        gen = self.read_data(self.start_idx, self.stop_idx)
        for (xdf, mdf, idx_label) in gen:
            data.append(xdf)
            mask.append(xdf)
        print(f"label {self.file_label_dict[self.current_file]}, "
        f"file <{self.current_file.split('/')[-1]}>, read {len(data)-len(labels)} events")
        label = self.file_label_dict[self.current_file]
        labels = np.append(labels, np.full(shape=len(data)-len(labels), fill_value=label, dtype=np.int))
        data = np.array(data)
        mask = np.array(mask)
        return data, mask, labels

    def __iter__(self):
        "Provide iterator capabilities to the class"
        return self

    def __next__(self):
        "Provide generator capabilities to the class"
        return self.next_mix_files()


    def check_file(self):
        "This function allows to set self.start_idx, self.stop_idx, and to change the file to be read if necessary"
        if self.evts != -1 and self.stop_idx > self.evts:
            if self.stop_idx - self.evts < self.chunk_size:
                self.stop_idx = self.evts
                return False
            # we finished reading all the self.evts events
            else:
                self.start_idx = 0
                self.stop_idx = self.chunk_size
                print("# we finished reading all the self.evts events")
                raise StopIteration
        if self.reader_counter[self.current_file] == self.reader[self.current_file].nrows:
            # if we exceed number of events in a file we discard it
            if self.verbose:
                msg = "\n# discard {} since we read {} out of {} events"\
                        .format(self.current_file, \
                        self.reader_counter[self.current_file], self.reader[self.current_file].nrows)
                print(msg)
            self.files.remove(self.current_file)
            self.file_label_dict.pop(self.current_file)
            if self.files:
                self.current_file = self.files[0]
                self.stop_idx = self.start_idx + self.chunk_size
                return True
            else:
                print("# no more files to read")
                raise StopIteration
        if self.reader_counter[self.current_file] + self.chunk_size > self.reader[self.current_file].nrows:
            self.stop_idx = self.start_idx + self.reader[self.current_file].nrows - self.reader_counter[self.current_file]
            return False
        else:
            return False

    def read_data(self, start=0, stop=100):
        "Helper function to read ROOT data via uproot reader"
        msg = "\nread chunk [{}:{}] from {}"\
                .format(self.start_idx, self.stop_idx-1, self.current_file)
        if self.verbose:
            print(msg)
        current_file = self.current_file
        reader = self.reader[current_file]
        reader.load_specs(self.gname)
        if stop == -1:
            for _ in range(reader.nrows):
                xdf, mask, idx_label = reader.next()
                yield (xdf, mask, idx_label)
            read_evts = reader.nrows
        else:
            for _ in range(self.start_idx, self.stop_idx):
                xdf, mask, idx_label = reader.next()
                yield (xdf, mask, idx_label)
            read_evts = self.stop_idx-self.start_idx
        # update how many events we read from current file
        self.reader_counter[self.current_file] += read_evts
        # advance start and stop indecies
        self.start_idx = self.stop_idx
        self.stop_idx = self.start_idx + self.chunk_size
        if self.verbose:
            nevts = self.reader_counter[self.current_file]
            msg = "\ntotal read {} evts from {}".format(nevts, current_file)
            #print(msg)
    
    def write_global_specs(self):
        if not os.path.isfile(self.gname):
            if self.verbose:
                print("write {}".format(self.gname))
            with open(self.gname, 'w') as ostream:
                out = {'jdim': self.jdim, 'minv': self.minv, 'maxv': self.maxv,\
                    'fkeys': self.fkeys, 'jkeys': self.jkeys, 'nans': self.nans,\
                        'events': self.events}
                ostream.write(json.dumps(out))

    def global_specs (self, fname, reader):
        "Function to build specs for the whole set of root files"
        if reader.preproc:
            if reader.evts:
                self.events[fname] = reader.evts[fname]
                self.events['total'] += self.events[fname]
                print("# %s total entries, %s total events after cut, (%s-flat, %s-jagged) branches, %s attrs\n" \
                      % (reader.nrows, self.events[fname], len(reader.flat_keys()), len(reader.jagged_keys()), reader.shape))
            else:
                print("--- Computing the number of events which satisfies the cuts on the whole file ---")
                global_timing = time.time()
                self.events[fname] = global_cut(reader.tree, reader.flat, reader.flat_preproc, reader.jagged, \
                                                reader.jagged_all, reader.jagged_any, reader.new_branch, reader.new_flat_cut, \
                                                reader.new_jagged_cut, reader.aliases_string, reader.total_key, reader)
                print("# %s total entries, %s total events after cut, (%s-flat, %s-jagged) branches, %s attrs" \
                % (reader.nrows, self.events[fname], len(reader.flat_keys()), len(reader.jagged_keys()), reader.shape))
                print("# total time elapsed: {} sec\n".format(round(time.time()-global_timing, 3)))
                self.events['total'] += self.events[fname]
        else:
            self.events[fname] = reader.nrows
            self.events['total'] += reader.nrows

        if fname == self.files[0]:
            self.jdim = reader.jdim
            self.minv = reader.minv
            self.maxv = reader.maxv
            self.fkeys = reader.fkeys
            self.jkeys = reader.jkeys
            if len(self.files) == 1:
                self.write_global_specs()
        else:
            for key in self.maxv.keys():
                if reader.maxv[key] > self.maxv[key]:
                    self.maxv[key] = reader.maxv[key]
                if reader.minv[key] < self.minv[key]:
                    self.minv[key] = reader.minv[key]
                if key in self.jkeys:
                    if reader.jdim[key] > self.jdim[key]:
                        self.jdim[key] = reader.jdim[key]
            if fname == self.files[-1]:
                for key in self.maxv.keys():
                    if self.minv[key] == self.maxv[key]:
                        self.nans[key] = self.maxv[key]
                    else:
                        self.nans[key] = (0-self.minv[key])/(self.maxv[key]-self.minv[key])
                self.write_global_specs()
