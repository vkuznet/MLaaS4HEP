from __future__ import print_function, division, absolute_import

# system modules
import os
import sys
import time
import json
import random
import argparse
import traceback
from itertools import takewhile, repeat

# for managing compressed files
import gzip
try:
    import bz2
except ImportError:
    pass

# numpy modules
import numpy as np

# numba
try:
    from numba import jit
except ImportError:
    def jit(f):
        "Simple decorator which calls underlying function"
        def new_f():
            "Action function"
            f()
        return new_f

# histogrammar
try:
    import histogrammar as hg
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt
except ImportError:
    hg = None

def timestamp(msg='MLaaS4HEP'):
    "Return timestamp in pre-defined format"
    tst = time.localtime()
    tstamp = time.strftime('[%d/%b/%Y:%H:%M:%S]', tst)
    return '%s %s %s' % (msg.strip(), tstamp, time.mktime(tst))

def load_code(mfile, fname):
    """
    Load function from given python module (file)

    :param mfile: the python module/file name which provides fname
    :param fname: name of the function from mfile
    """
    mname = mfile.split('.py')[0].replace('/', '.')
    try:
        mod = __import__(mname, fromlist=['model'])
        func = getattr(mod, fname)
        print("load {} {} {}".format(mfile, func, func.__doc__))
        return func
    except ImportError:
        traceback.print_exc()
        msg = "Please provide file name with 'def %s' implementation" % fname
        msg += "\nThe file should be available in PYTHONPATH"
        print(msg)
        raise

def nrows(filename):
    """
    Return total number of rows in given file, see
    https://stackoverflow.com/questions/845058/how-to-get-line-count-cheaply-in-python
    """
    with fopen(filename, 'rb') as f:
        if  sys.version.startswith('3.'):
            bufgen = takewhile(lambda x: x, (f.raw.read(1024*1024) for _ in repeat(None)))
        else:
            bufgen = takewhile(lambda x: x, (f.read(1024*1024) for _ in repeat(None)))
        return sum( buf.count(b'\n') for buf in bufgen )

def dump_histograms(hdict, hgkeys):
    "Helper function to dump histograms"
    if not hg:
        return
    for key in hgkeys:
        make_plot(hdict['%s_orig' % key], '%s_orig' % key)
        make_plot(hdict['%s_norm' % key], '%s_norm' % key)

def make_plot(hist, name):
    "Helper function to make histogram"
    pdir = os.path.join(os.getcwd(), 'pdfs')
    try:
        os.makedirs(pdir)
    except OSError:
        pass
    fname = os.path.join(pdir, '%s.pdf' % name)
    pdf = PdfPages(fname)
    fig = plt.figure()
    hist.plot.matplotlib(name=name)
    pdf.savefig(fig)
    plt.close(fig)
    pdf.close()

def mem_usage(vmem0, swap0, vmem1, swap1, msg=None):
    "helper function to show memory usage"
    if msg:
        print(msg)
    mbyte = 10**6
    vmem_used = (vmem1.used-vmem0.used)/mbyte
    swap_used = (swap1.used-swap0.used)/mbyte
    print("VMEM used: %s (MB) SWAP used: %s (MB)" % (vmem_used, swap_used))

def performance(nevts, tree, data, startTime, endTime, msg=""):
    "helper function to show performance metrics of data read from a given tree"
    try:
        nbytes = sum(x.content.nbytes + x.stops.nbytes \
                if isinstance(x, JaggedArray) \
                else x.nbytes for x in data.values())
        print("# %s entries, %s %sbranches, %s MB, %s sec, %s MB/sec, %s kHz" % \
                (
            nevts,
            len(data),
            msg,
            nbytes / 1024**2,
            endTime - startTime,
            nbytes / 1024**2 / (endTime - startTime),
            tree.numentries / (endTime - startTime) / 1000))
    except Exception as exc:
        print(str(exc))

def steps(total, size):
    "Return list of steps within total number given events"
    step = int(float(total)/float(size))
    chunk = []
    for idx in range(total):
        if len(chunk) == size:
            yield chunk
            chunk = []
        chunk.append(idx)
    if len(chunk) > 0:
        yield chunk


def fopen(fin, mode='r'):
    "Return file descriptor for given file"
    if  fin.endswith('.gz'):
        stream = gzip.open(fin, mode)
    elif  fin.endswith('.bz2'):
        stream = bz2.BZ2File(fin, mode)
    else:
        stream = open(fin, mode)
    return stream

def file_type(fin):
    "Return file type of given object"
    if isinstance(fin, list):
        fin = fin[0]
    fin = fin.lower()
    for ext in ['root', 'avro']:
        if fin.endswith(ext):
            return ext
    for ext in ['json', 'csv']:
        if fin.endswith(ext) or fin.endswith('%s.gz' % ext) or fin.endswith('%s.bz2' % ext):
            return ext

