#-*- coding: utf-8 -*-
#pylint: disable=R0913
"""
General set of utilities used by MLaaS4HEP framework.
"""
from __future__ import print_function, division, absolute_import

# system modules
import os
import time
import traceback
import numpy as np
from itertools import takewhile, repeat

# for managing compressed files
import gzip
try:
    import bz2
except ImportError:
    pass

# uproot
try:
    import uproot
    import awkward as ak
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
    with fopen(filename, 'rb') as fdsc:
        bufgen = takewhile(lambda x: x, (fdsc.read(1024*1024) for _ in repeat(None)))
        return sum([buf.count(b'\n') for buf in bufgen])

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

def performance(nevts, tree, data, start_time, end_time, msg=""):
    "helper function to show performance metrics of data read from a given tree"
    try:
        nbytes = sum(x.content.nbytes + x.stops.nbytes \
                if isinstance(x, uproot.AsJagged) \
                else x.nbytes for x in data.values())
        print("# %s entries, %s %sbranches, %s MB, %s sec, %s MB/sec, %s kHz" % \
                ( \
            nevts, \
            len(data), \
            msg, \
            nbytes / 1024**2, \
            end_time - start_time, \
            nbytes / 1024**2 / (end_time - start_time), \
            nevts / (end_time - start_time) / 1000))
    except Exception as exc:
        print(str(exc))

def steps(total, size):
    "Return list of steps within total number given events"
    chunk = []
    for idx in range(total):
        if len(chunk) == size:
            yield chunk
            chunk = []
        chunk.append(idx)
    if chunk:
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
    return None

def flat_handling(cut):
    "Handling the flat cut(s) in  preproc.json"
    cut_string = ''
    branch_name = ''
    global_string = ''
    op = ["==", ">", "!=", "<"]

    for i in range(0, len(cut)):

        for symbol in op:
            if cut[i].find(symbol) != -1:
                x = cut[i].partition(symbol)
                x_name, x_cond, x_num = x[0], x[1], x[2]
            else:
                continue

            if (global_string == ''):
                global_string += str('(reader.tree["' + x_name + '"].array()' + x_cond + x_num + ')')
            else:
                global_string += str(' & (reader.tree["' + x_name + '"].array()' + x_cond + x_num + ')')

        if i == 0:
            branch_name = x[0]
            cut_string += str('(' + cut[i] + ')')
        else:
            cut_string += str(' & (' + cut[i] + ')')

    flat_cut = [cut_string, global_string, branch_name]

    return flat_cut

def jagged_handling(jagged):

    all_str = 'all'
    cut_all = ''
    cut_any = ''
    global_all = ''
    global_any = ''
    op = ["==", ">", "!=", '<']
    jagged_all = []
    jagged_any = []
    cut_str = ''
    for i in range(0, len(jagged)):
        cut_str += str('(' + jagged[i][0] + ')' + ' using ' + '"' + jagged[i][1] + '"' + '\n')

        for symbol in op:
            if jagged[i][0].find(symbol) != -1:
                x = jagged[i][0].partition(symbol)
                x_name, x_cond, x_num = x[0], x[1], x[2]
            else:
                continue

            if jagged[i][1] == all_str:
                if cut_all == '':
                    cut_all += str('(batch["' + x_name + '"]' + x_cond + x_num + ')')
                else:
                    cut_all += str(' & (batch["' + x_name + '"]' + x_cond + x_num + ')')
                if global_all == '':
                    global_all += str('(tree["' + x_name + '"].array()' + x_cond + x_num + ')')
                else:
                    global_all += str(' & (tree["' + x_name + '"].array()' + x_cond + x_num + ')')

            else:
                if cut_any == '':
                    cut_any += str('(batch["' + x_name + '"]' + x_cond + x_num + ')')
                else:
                    cut_any += str(' & (batch["' + x_name + '"]' + x_cond + x_num + ')')

                if global_any == '':
                    global_any += str('(tree["' + x_name + '"].array()' + x_cond + x_num + ')')
                else:
                    global_any += str(' & (tree["' + x_name + '"].array()' + x_cond + x_num + ')')

    if cut_all != '':
        jagged_all.append(cut_all)
    if global_all != '':
        jagged_all.append(global_all)
    if cut_any != '':
        jagged_any.append(cut_any)
    if global_any != '':
        jagged_any.append(global_any)

    return jagged_all, jagged_any

def new_branch_handling(tree, new_branch, new_flat_cut, new_jagged_cut, nbranch, to_remove):

    op_branch = []
    type_branch = []
    new_branch_remove = []

    for key in new_branch.keys():

        #nome nuovo branch
        nbranch.append(key)
        #come è definito
        op_branch.append(new_branch[key]['def'])
        #se è jagged o flat
        type_branch.append(new_branch[key]['type'])

        #se il new branch è jagged, metto in new_j_cut una lista contenente [taglio, all/any]
        if new_branch[key]['cut']:
            if new_branch[key]['type'] == "jagged":
                _ = []
                _.extend([new_branch[key]['cut'], new_branch[key]['cut_type']])
                new_jagged_cut.append(_)

            #altrimenti appendo a new_f_cut  i tagli semplici
            else:
                new_flat_cut.append(new_branch[key]['cut'])

        #metto dentro la lista new_branch_remove delle liste contenenti [nome, true/false]
        rem = []
        rem.extend([key, new_branch[key]['remove']])
        new_branch_remove.append(rem)

    to_remove += new_branch_remove


    aliases_string = aliases_func(nbranch, op_branch)

    total_key = tree.keys() + nbranch

    return nbranch, new_flat_cut, new_jagged_cut, aliases_string, to_remove

def aliases_func(nbranch, opbranch):

    aliases_string = ''

    for i in range(len(nbranch)):
        if len(nbranch)>1:
            if i == 0:
                aliases_string += str("{'" + nbranch[i] + "': " + "'" + opbranch[i] + "'")
            elif i == len(nbranch)-1:
                aliases_string += str(", '" + nbranch[i] + "': " + "'" + opbranch[i] + "'}")
            else:
                aliases_string += str(", '" + nbranch[i] + "': " + "'" + opbranch[i] + "'")
        else:
            aliases_string += str("{'" + nbranch[i] + "': " + "'" + opbranch[i] + "'}")

    return aliases_string


def gen_preproc(tree, nevts, flat, jagged, flat_preproc, aliases_string, new_branch, new_flat_cut, total_key):

    """A seconda del taglio selezionato, returno il giusto generatore gen"""

    if (flat != {}) & (jagged == {}):
        if new_branch != {}:
            gen = tree.iterate(total_key, cut=flat_preproc[0], step_size=nevts, aliases=eval(aliases_string), library='ak')
        else:
            gen = tree.iterate(cut=flat_preproc[0], step_size=nevts, library='np')

    elif (flat != {}) & (jagged != {}):
        if new_branch != {}:
            gen = tree.iterate(total_key, cut=flat_preproc[0], step_size=nevts, aliases=eval(aliases_string), library='ak')
        else:
            gen = tree.iterate(cut=flat_preproc[0], step_size=nevts, library='ak')
    else:
        if new_branch != {}:
            if new_flat_cut:
                gen = tree.iterate(total_key, cut=flat_preproc[0], step_size= nevts, aliases=eval(aliases_string), library='ak')
            else:
                gen = tree.iterate(total_key, step_size= nevts, aliases=eval(aliases_string), library='ak')
        else:
            gen = tree.iterate(step_size= nevts, library='ak')

    return gen

def cutted_next(gen, flat_preproc, jagged, jagged_all, jagged_any, new_branch, new_jagged_cut):

    if jagged:
        batch = next(gen)
        cutted_batch, cutted_evts = if_jagged_cut(batch, jagged_all, jagged_any)
        branches = dict(zip(ak.fields(cutted_batch), ak.unzip(cutted_batch)))
    else:
        if new_branch:
            batch = next(gen)
            if new_jagged_cut:
                cutted_batch, cutted_evts = if_jagged_cut(batch, jagged_all, jagged_any)
            else:
                cutted_evts = len(batch)

            branches = dict(zip(ak.fields(batch), ak.unzip(batch)))
        else:
            branches = next(gen)
            cutted_evts = len(branches[flat_preproc[2]])

    return branches, cutted_evts

def if_jagged_cut(batch, jagged_all, jagged_any):
    if jagged_all:
        if jagged_any:
            cutted_batch = batch[(ak.all(eval(jagged_all[0]), axis=1)) & ak.any(eval(jagged_any[0]), axis=1)]
            cutted_evts = len(cutted_batch)

        else:
            cutted_batch = batch[(ak.all(eval(jagged_all[0]), axis=1))]
            cutted_evts = len(cutted_batch)

    else:
        cutted_batch = batch[ak.any(eval(jagged_any[0]), axis=1)]
        cutted_evts = len(cutted_batch)

    return cutted_batch, cutted_evts

def global_cut(tree, flat, flat_preproc, jagged, jagged_all, jagged_any, new_branch, new_flat_cut, new_jagged_cut, aliases_string, total_key, reader):

    tot_cutted_evts = 0
    if (flat != {}) & (jagged != {}):
        if new_branch:
            new_gen = tree.iterate(total_key, cut=flat_preproc[0], step_size=100000, aliases=eval(aliases_string), library='ak')
            tot_cutted_evts = new_jagged_global(new_gen, jagged_all, jagged_any, reader)
        else:
            tot_cutted_evts = jagged_global(tree, flat_preproc, jagged_all, jagged_any, reader)

    #se ci sono jagged ma non flat
    elif (jagged != {}) & (flat == {}):
        if new_branch:
            if new_flat_cut:
                new_gen = tree.iterate(total_key, cut=flat_preproc[0], step_size=100000, aliases=eval(aliases_string), library='ak')
            else:
                new_gen = tree.iterate(total_key, step_size=100000, aliases=eval(aliases_string), library='ak')
            tot_cutted_evts = new_jagged_global(new_gen, jagged_all, jagged_any, reader)

        else:
            tot_cutted_evts = jagged_global(tree, flat_preproc, jagged_all, jagged_any, reader)

    #se ci sono flat ma non jagged
    elif (flat != {}) & (jagged == {}):
        if new_branch:
            new_gen = tree.iterate(total_key, cut=flat_preproc[0], step_size=100000, aliases=eval(aliases_string), library='ak')
            if new_jagged_cut:
                tot_cutted_evts = new_jagged_global(new_gen, jagged_all, jagged_any, reader)
            else:
                tot_cutted_evts = flat_global(new_gen, reader)
        else:
            tot_cutted_evts = np.count_nonzero((eval(flat_preproc[1])))

    #no flat e no jagged
    else:
        if new_branch:
            if new_flat_cut:
                new_gen = tree.iterate(total_key, cut=flat_preproc[0], step_size=100000, aliases=eval(aliases_string), library='ak')
                if new_jagged_cut:
                    tot_cutted_evts = new_jagged_global(new_gen, jagged_all, jagged_any, reader)
                else:
                    tot_cutted_evts = flat_global(new_gen, reader)

            else:
                new_gen = tree.iterate(total_key, step_size=100000, aliases=eval(aliases_string), library='ak')
                if new_jagged_cut:
                    tot_cutted_evts = new_jagged_global(new_gen, jagged_all, jagged_any, reader)
                else:
                    #no cut, only new branch definition
                    tot_cutted_evts = tree.num_entries

        else:
            print("EMPTY PREPROC")

    return(tot_cutted_evts)


def new_jagged_global(new_gen, jagged_all, jagged_any, reader):

    cutted_counter = 0
    n_batch = 0
    for batch in new_gen:
        len_batch = len(batch)
        if (jagged_all != []) & (jagged_any != []):
            cutted_batch = batch[(ak.all(eval(jagged_all[0]), axis=1)) & (ak.any(eval(jagged_any[0]), axis=1))]
        elif (jagged_all != []) & (jagged_any == []):
            cutted_batch = batch[ak.all(eval(jagged_all[0]), axis=1)]
        else:
            cutted_batch = batch[ak.any(eval(jagged_any[0]), axis=1)]
        cutted_evts = len(cutted_batch)
        #print("{} - {} = {} eventi non superano il taglio".format(len_batch, cutted_evts, len_batch - cutted_evts))
        cutted_counter += cutted_evts
        n_batch += 1
        #print("batch numero {}".format(n_batch))
    tot_cutted_evts = cutted_counter

    return tot_cutted_evts

def jagged_global(tree, flat_preproc, jagged_all, jagged_any, reader):

    if flat_preproc:
        if (jagged_all != []) & (jagged_any != []):
            tot_cutted_evts = np.count_nonzero((eval(flat_preproc[1])) & ak.all(eval(jagged_all[1]), axis=1) & ak.any(eval(jagged_any[1]), axis=1))
        elif (jagged_all != []) & (jagged_any == []):
            tot_cutted_evts = np.count_nonzero((eval(flat_preproc[1])) & ak.all(eval(jagged_all[1]), axis=1))
        else:
            tot_cutted_evts = np.count_nonzero((eval(flat_preproc[1])) & ak.any(eval(jagged_any[1]), axis=1))
    else:
        if (jagged_all != []) & (jagged_any != []):
            tot_cutted_evts = np.count_nonzero(ak.all(eval(jagged_all[1]), axis=1) & ak.any(eval(jagged_any[1]), axis=1))
        elif (jagged_all != []) & (jagged_any == []):
            tot_cutted_evts = np.count_nonzero(ak.all(eval(jagged_all[1]), axis=1))
        else:
            tot_cutted_evts = np.count_nonzero(ak.any(eval(jagged_any[1]), axis=1))

    return tot_cutted_evts

def flat_global(new_gen, reader):

    cutted_counter = 0
    n_batch = 0
    for batch in new_gen:
        len_batch = len(batch)
        cutted_counter += len_batch
        n_batch += 1
        #print("batch numero {}".format(n_batch))
    tot_cutted_evts = cutted_counter

    return tot_cutted_evts

def print_cut(preproc):

    flat = preproc['flat_cut']
    jagged = preproc['jagged_cut']
    new = preproc['new_branch']

    print('\n')

    if flat != {}:
        print("Cut(s) provided on flat branch(es):")
        for key in flat.keys():
            print(flat[key]['cut'])
    else:
        print('No cut on flat branches was provided')

    if jagged != {}:
        print("Cut(s) provided on jagged branch(es):")
        for key in jagged.keys():
            print('{} using "{}"'.format(jagged[key]['cut'], jagged[key]['cut_type']))
    else:
        print('No cut on jagged branches was provided')

    if new != {}:
        print("Definition(s) of new branch(es): ")
        for key in new.keys():
            print('{}: {}'.format(key, new[key]['def']))

        print("Cut(s) provided on new branch(es):")
        for key in new.keys():
            if new[key]['cut']:
                if new[key]['type'] == "jagged":
                    print('Jagged: {} using "{}"'.format(new[key]['cut'], new[key]['cut_type']))
                else:
                    print('Flat: {}'.format(new[key]['cut']))
            else:
                print('No cut provided for the new branch: {}'.format(key))
    else:
        print("New branch was not provided")
    print('\n')

    return "pippo"
