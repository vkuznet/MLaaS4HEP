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
        #print("load {} {} {}".format(mfile, func, func.__doc__))
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

def performance(nevts, tree, data, start_time, end_time, cutted_evts=None, msg=""):
    "helper function to show performance metrics of data read from a given tree"
    try:
        if cutted_evts:
            nbytes = sum(x.content.nbytes + x.stops.nbytes \
                             if isinstance(x, uproot.AsJagged) \
                             else x.nbytes for x in data.values())
            print("# %s entries, %s events after cut, %s %sbranches, %s MB, %s sec, %s MB/sec, %s kHz" % \
                  ( \
                      nevts, \
                      cutted_evts, \
                      len(data), \
                      msg, \
                      round(nbytes / 1024**2, 3), \
                      round(end_time - start_time, 3), \
                      round(nbytes / 1024**2 / (end_time - start_time), 3), \
                      round(nevts / (end_time - start_time) / 1000, 3)))
        else:
            nbytes = sum(x.content.nbytes + x.stops.nbytes \
                             if isinstance(x, uproot.AsJagged) \
                             else x.nbytes for x in data.values())
            print("# %s entries, %s %sbranches, %s MB, %s sec, %s MB/sec, %s kHz" % \
                  ( \
                      nevts, \
                      len(data), \
                      msg, \
                      round(nbytes / 1024**2, 3), \
                      round(end_time - start_time, 3), \
                      round(nbytes / 1024**2 / (end_time - start_time), 3), \
                      round(nevts / (end_time - start_time) / 1000, 3)))
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

    "Handling the flat cut(s) in preproc.json"

    cut_string = ''
    branch_name = ''
    global_string = ''
    op = ["==", ">=", "<=", "!=", ">", "<"]

    for i in range(0, len(cut)):

        for symbols in op:
            if cut[i].find(symbols) != -1:
                #print(cut[i])
                x = cut[i].partition(symbols)
                x_name, x_cond, x_num = x[0], x[1], x[2]
                #print(x_name, x_cond, x_num)

                x_name_lower = x_name.lower()
                name_or_number = x_name_lower.islower()
                if not name_or_number:
                    for symbols in op:
                        if x_num.find(symbols) != -1:
                            y = x_num.partition(symbols)
                            y_name, y_cond, y_num = y[0], y[1], y[2]
                            #print(y_name, y_cond, y_num)
                            break
                    if i == 0:
                        branch_name = y_name
                        if x_cond == '<=':
                            cut_string += str('(' + y_name + '>=' + x_name + ') & (' + y_name + y_cond + y_num + ')')
                        else:
                            cut_string += str('(' + y_name + '>' + x_name + ') & (' + y_name + y_cond + y_num + ')')
                    else:
                        if x_cond == '<=':
                            cut_string += str(' & (' + y_name + '>=' + x_name + ') & (' + y_name + y_cond + y_num + ')')
                        else:
                            cut_string += str(' & (' + y_name + '>' + x_name + ') & (' + y_name + y_cond + y_num + ')')

                    if x_cond == '<=':
                        if global_string == '':
                            global_string += str('(tree["' + y_name + '"].array()>=' + x_name + ') & (tree["' + y_name + '"].array()' + y_cond + y_num + ')')
                        else:
                            global_string += str(' & (tree["' + y_name + '"].array()>=' + x_name + ') & (tree["' + y_name + '"].array()' + y_cond + y_num + ')')
                    else:
                        if global_string == '':
                            global_string += str('(tree["' + y_name + '"].array()>' + x_name + ') & (tree["' + y_name + '"].array()' + y_cond + y_num + ')')
                        else:
                            global_string += str(' & (tree["' + y_name + '"].array()>' + x_name + ') & (tree["' + y_name + '"].array()' + y_cond + y_num + ')')

                else:
                    if i == 0:
                        branch_name = x_name
                        cut_string += str('(' + cut[i] + ')')
                    else:
                        cut_string += str(' & (' + cut[i] + ')')

                    if global_string == '':
                        global_string += str('(tree["' + x_name + '"].array()' + x_cond + x_num + ')')
                    else:
                        global_string += str(' & (tree["' + x_name + '"].array()' + x_cond + x_num + ')')
                break

            else:
                continue

    flat_cut = [cut_string, global_string, branch_name]

    return flat_cut


def jagged_handling(jagged):

    "Handling the jagged cut(s) in preproc.json"

    all_str = 'all'
    cut_all = ''
    cut_any = ''
    global_all = ''
    global_any = ''
    op = ["==", ">", "!=", '<']
    jagged_all = []
    jagged_any = []

    for i in range(0, len(jagged)):
        for symbol in op:
            if jagged[i][0].find(symbol) != -1:
                x = jagged[i][0].partition(symbol)
                x_name, x_cond, x_num = x[0], x[1], x[2]
            else:
                continue

            if jagged[i][1] == all_str:
                if cut_all == '':
                    cut_all += str('(ak.all(batch["' + x_name + '"]' + x_cond + x_num + ', axis=1))')
                else:
                    cut_all += str(' & (ak.all(batch["' + x_name + '"]' + x_cond + x_num + ', axis=1))')
                if global_all == '':
                    global_all += str('(ak.all(tree["' + x_name + '"].array()' + x_cond + x_num + ', axis=1))')
                else:
                    global_all += str(' & (ak.all(tree["' + x_name + '"].array()' + x_cond + x_num + ', axis=1))')

            else:
                if cut_any == '':
                    cut_any += str('(ak.any(batch["' + x_name + '"]' + x_cond + x_num + ', axis=1))')
                else:
                    cut_any += str(' & (ak.any(batch["' + x_name + '"]' + x_cond + x_num + ', axis=1))')

                if global_any == '':
                    global_any += str('(ak.any(tree["' + x_name + '"].array()' + x_cond + x_num + ', axis=1))')
                else:
                    global_any += str(' & (ak.any(tree["' + x_name + '"].array()' + x_cond + x_num + ', axis=1))')

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

    "Handling the new branches in preproc.json"

    op_branch = []
    type_branch = []
    new_branch_remove = []

    for key in new_branch.keys():

        nbranch.append(key)
        op_branch.append(new_branch[key]['def'])
        type_branch.append(new_branch[key]['type'])

        for elem in new_branch[key]:
            if elem.startswith("cut"):
                if new_branch[key]['type'] == "jagged":
                    new_branch[key][elem] = [x.replace(" ", "") for x in new_branch[key][elem]]
                    new_jagged_cut.append(new_branch[key][elem])

                else:
                    new_branch[key][elem] = new_branch[key][elem].replace(" ", "")
                    new_flat_cut.append(new_branch[key][elem])

            if elem.startswith("keys_to_"):
                rem_2 = []
                for elem in new_branch[key]['keys_to_remove']:
                    _ = []
                    _.extend([elem, "True"])
                    rem_2.append(_)

                for elem in rem_2:
                    if elem not in to_remove:
                        new_branch_remove.append(elem)
                    else:
                        continue

        rem = []
        rem.extend([key, new_branch[key]['remove']])
        new_branch_remove.append(rem)

    to_remove += new_branch_remove

    aliases_string = aliases_func(nbranch, op_branch)

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
            cutted_batch = batch[eval(jagged_all[0]) & eval(jagged_any[0])]
            cutted_evts = len(cutted_batch)

        else:
            cutted_batch = batch[eval(jagged_all[0])]
            cutted_evts = len(cutted_batch)

    else:
        cutted_batch = batch[eval(jagged_any[0])]
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

    #jagged and not flat
    elif (jagged != {}) & (flat == {}):
        if new_branch:
            if new_flat_cut:
                new_gen = tree.iterate(total_key, cut=flat_preproc[0], step_size=100000, aliases=eval(aliases_string), library='ak')
            else:
                new_gen = tree.iterate(total_key, step_size=100000, aliases=eval(aliases_string), library='ak')
            tot_cutted_evts = new_jagged_global(new_gen, jagged_all, jagged_any, reader)

        else:
            tot_cutted_evts = jagged_global(tree, flat_preproc, jagged_all, jagged_any, reader)

    #flat and not jagged
    elif (flat != {}) & (jagged == {}):
        if new_branch:
            new_gen = tree.iterate(total_key, cut=flat_preproc[0], step_size=100000, aliases=eval(aliases_string), library='ak')
            if new_jagged_cut:
                tot_cutted_evts = new_jagged_global(new_gen, jagged_all, jagged_any, reader)
            else:
                tot_cutted_evts = flat_global(new_gen, reader)
        else:
            tot_cutted_evts = np.count_nonzero((eval(flat_preproc[1])))

    #no flat and no jagged
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

    for batch in new_gen:
        if (jagged_all != []) & (jagged_any != []):
            cutted_batch = batch[eval(jagged_all[0]) & eval(jagged_any[0])]
        elif (jagged_all != []) & (jagged_any == []):
            cutted_batch = batch[eval(jagged_all[0])]
        else:
            cutted_batch = batch[eval(jagged_any[0])]
        cutted_evts = len(cutted_batch)

        cutted_counter += cutted_evts

    tot_cutted_evts = cutted_counter

    return tot_cutted_evts

def jagged_global(tree, flat_preproc, jagged_all, jagged_any, reader):

    if flat_preproc != []:
        if (jagged_all != []) & (jagged_any != []):
            tot_cutted_evts = np.count_nonzero(eval(flat_preproc[1]) & eval(jagged_all[1]) & eval(jagged_any[1]))
        elif (jagged_all != []) & (jagged_any == []):
            tot_cutted_evts = np.count_nonzero(eval(flat_preproc[1]) & eval(jagged_all[1]))
        else:
            tot_cutted_evts = np.count_nonzero((eval(flat_preproc[1])) & eval(jagged_any[1]))
    else:
        if (jagged_all != []) & (jagged_any != []):
            tot_cutted_evts = np.count_nonzero(eval(jagged_all[1]) & eval(jagged_any[1]))
        elif (jagged_all != []) & (jagged_any == []):
            tot_cutted_evts = np.count_nonzero(eval(jagged_all[1]))
        else:
            tot_cutted_evts = np.count_nonzero(eval(jagged_any[1]))

    return tot_cutted_evts

def flat_global(new_gen, reader):

    cutted_counter = 0
    for batch in new_gen:
        len_batch = len(batch)
        cutted_counter += len_batch
    tot_cutted_evts = cutted_counter

    return tot_cutted_evts

def print_cut(preproc):

    new_bool = False
    flat_bool = False
    jagged_bool = False

    for elem in preproc.keys():
        if elem.startswith("new_"):
            new = preproc['new_branch']
            new_bool = True

        if elem.startswith("flat_"):
            flat = preproc['flat_cut']
            flat_bool = True

        if elem.startswith("jagged_"):
            jagged = preproc['jagged_cut']
            jagged_bool = True

    if new_bool == False:
        new = {}
    if flat_bool == False:
        flat = {}
    if jagged_bool == False:
        jagged = {}


    if flat != {}:
        print("# Cut(s) provided on flat branch(es):")
        for key in flat.keys():
            for elem in flat[key]:
                if elem.startswith("cut"):
                    print(flat[key][elem])
                else:
                    break


    if jagged != {}:
        print("# Cut(s) provided on jagged branch(es):")
        for key in jagged.keys():
            for elem in jagged[key]:
                if elem.startswith("cut"):
                    print('{} using "{}"'.format(jagged[key][elem][0], jagged[key][elem][1]))
                else:
                    break


    if new != {}:
        print("# Definition(s) of new branch(es): ")
        for key in new.keys():
            print('{}: {}'.format(key, new[key]['def']))
        print("# Cut(s) on new branch(es):")
        for key in new.keys():
            gatsu = 0
            for elem in new[key]:
                if elem.startswith("cut"):
                    if new[key]['type'] == "jagged":
                        print('{} using "{}"'.format(new[key][elem][0], new[key][elem][1]))
                    else:
                        print('{}'.format(new[key][elem]))
                    gatsu += 1
                else:
                    continue
            if gatsu == 0:
                print("No cut(s) on {}".format(key))

        print('\n')


