""" DESI utilitary functions """
import numpy as np
from pycorr import TwoPointCorrelationFunction
import os 



def read_pc(path_data = '/global/cfs/cdirs/desi/survey/catalogs/edav1/xi/sv3/',
            stat = 'rppi',
            tracer = 'ELG',
            region = 'NScomb',
            zlim = '0.8_1.6',
            njk = 128,
            nran = 18,
            weights = 'default_angular_bitwise_FKP',
            binning = 'log',
            split='_split20',
            fn=None,
            n_rebin = 1,
            n_rebin_post_cut=1,
            bin_inf=None,
            bin_sup=None):
    if fn is None:
        name = f'allcounts_{tracer}_{region}_{zlim}_{weights}_{binning}_njack{njk}_nran{nran}{split}.npy'
        fn = os.path.join(path_data, stat, name)
        return TwoPointCorrelationFunction.load(fn)[::n_rebin][bin_inf:bin_sup][::n_rebin_post_cut]
    else:
        return TwoPointCorrelationFunction.load(fn)[::n_rebin][bin_inf:bin_sup][::n_rebin_post_cut]