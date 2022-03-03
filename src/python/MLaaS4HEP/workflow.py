#!/usr/bin/env python
#-*- coding: utf-8 -*-
#pylint: disable=R0903

"""
File       : workflow.py
Author     : Valentin Kuznetsov <vkuznet AT gmail dot com>
MLaaS4HEP workflow module provides example how to read HEP files and train over them ML model
"""
from __future__ import print_function, division, absolute_import

# system modules
import os
import sys
import json
import argparse

# MLaaS4HEP modules
from models import train_model

class OptionParser(object):
    "OptionParser class for workflow arguments"
    def __init__(self):
        "User based option parser"
        self.parser = argparse.ArgumentParser(prog='PROG')
        self.parser.add_argument("--model", action="store", \
            dest="model", default=None, \
            help="Input model file, see ex_keras.py or ex_pytorch.py for examples")
        self.parser.add_argument("--preproc", action="store", \
            dest="preproc", default=None, \
            help="Input preprocessing file")
        self.parser.add_argument("--params", action="store", \
            dest="params", default="params.json", \
            help="Input model parameters (default params.json)")
        self.parser.add_argument("--specs", action="store", \
            dest="specs", default=None, help="Input specs file")
        self.parser.add_argument("--files", action="store", \
            dest="files", default='', \
            help="either input file with files names or comma separate list of files")
        self.parser.add_argument("--dtype", action="store", \
            dest="dtype", default=None, \
            help="specify data type of files: json, csv, parquet, avro, root")
        self.parser.add_argument("--labels", action="store", \
            dest="labels", default='', \
            help="either input file with labels names/ids or comma separate list of labels")
        self.parser.add_argument("--fout", action="store", \
            dest="fout", default=None, \
            help="output file name to save pre-trained model")

def main():
    "Main function"
    optmgr = OptionParser()
    opts = optmgr.parser.parse_args()
    if opts.params and os.path.exists(opts.params):
        params = json.load(open(opts.params))
    else:
        params = {}
    specs = json.load(open(opts.specs)) if opts.specs else None
    if not opts.files:
        print('No files is provided')
        sys.exit(1)
    if os.path.isfile(opts.files):
        files = [f.replace('\n', '') for f in open(opts.files).readlines() if not f.startswith('#')]
    else:
        files = opts.files.split(',')
    if not opts.labels:
        print('Please provide either label file or label name to look-up in data')
        sys.exit(1)
    if os.path.isfile(opts.labels):
        labels = [f.replace('\n', '') for f in open(opts.labels).readlines() \
                if not f.startswith('#')]
    elif opts.labels.find(',') != -1: # label list
        labels = opts.labels.split(',')
    else:
        labels = opts.labels # label name to look-up in data

    if opts.model:
        train_model(opts.model, files, labels, \
                preproc=opts.preproc, params=params, specs=specs, \
                fout=opts.fout, dtype=opts.dtype)
        return

if __name__ == '__main__':
    main()
