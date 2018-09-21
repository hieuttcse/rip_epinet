# Trung-Hieu Tran @ipvs
# 180921

from __future__ import print_function
import h5py
import os
import numpy as np


# Create a dataset with the first dim is expandable
def create_appendable_dset(fname,dset_name,data):
    dshape = data.shape
    maxshape = [None]*len(dshape)
    for i in range(1,len(dshape)):
        maxshape[i] = dshape[i]

    with h5py.File(fname,'a') as f:
         dset = f.create_dataset(dset_name,dshape,maxshape=maxshape,
                                 dtype=data.dtype,chunks=True)
         dset[:] = data

# write data into h5 dataset in an appendable way
# data should have the following form: B x D0 x D1 x ...
# The first dim is expandable dim, B is the number of new data will be
# written
def append_data(fname, dset_name, data):
    avail = True
    if not os.path.exists(fname):
        avail = False
    else:
        with h5py.File(fname,'r') as f:
            if not dset_name in f:
                avail = False
    if not avail:
        create_appendable_dset(fname,dset_name,data)
    else:
        with h5py.File(fname,'a') as f:
            dset = f[dset_name]
            dset.resize(dset.shape[0]+data.shape[0],axis=0)
            dset[-data.shape[0]:,...] = data
