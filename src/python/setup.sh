#!/bin/bash
if [ "`hostname`" == "vkair" ]; then
    # setup uproot
    #export PYTHONPATH=/Users/vk/CMS/RandD/usr/lib/python2.7/site-packages:$PYTHONPATH
    # setup pyxrootd
    #export PYTHONPATH=$PYTHONPATH:/Users/vk/CMS/RandD/xrootd-4.7.0/build/build/lib.macosx-10.13-x86_64-2.7
    source /opt/anaconda2/etc/profile.d/conda.sh
#    export PYTHONPATH=$PYTHONPATH:/opt/anaconda2/lib/python2.7/site-packages
else
    source /afs/cern.ch/user/v/valya/workspace/anaconda2/etc/profile.d/conda.sh
    conda activate myenv
fi
