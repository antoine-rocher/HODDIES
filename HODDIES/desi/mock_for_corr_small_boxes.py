from HODDIES import HOD
from pycorr import TwoPointCorrelationFunction, utils
import yaml
from HODDIES.utils import apply_rsd
import os
import glob
import argparse
import numpy as np
from matplotlib import pyplot as plt

def get_corr_small_boxes(zsim, corr, pimax=40, ells=[0,2], show=True):
    wp, xi = None, None
    if 'rppi' in corr:
        wp = [TwoPointCorrelationFunction.load(file)(pimax=pimax) for file in glob.glob(f'/global/cfs/cdirs/desi/users/arocher/Y1/2PCF_for_corr/Abcaus_small_boxes/z{zsim}/rppi/allcounts_AbacusSummit_small_c000_ph*.npy')]

    if 'smu' in corr:
        xi = [np.hstack(TwoPointCorrelationFunction.load(file)(ells=ells)) for file in glob.glob(f'/global/cfs/cdirs/desi/users/arocher/Y1/2PCF_for_corr/Abcaus_small_boxes/z{zsim}/smu/allcounts_AbacusSummit_small_c000_ph*.npy')]
        
    if wp is None:
        corr = utils.cov_to_corrcoef(np.cov(xi, rowvar=False, ddof=0))
    elif xi is None:
        corr = utils.cov_to_corrcoef(np.cov(wp, rowvar=False, ddof=0))
    else:
        corr = utils.cov_to_corrcoef(np.cov(np.hstack((wp,xi)), rowvar=False, ddof=0))
    if show:
        plt.imshow(corr)
    return corr

def plot_cov_measurement():

    fig, axx = plt.subplots(1,3, figsize=(12,4))
    ax = axx[0]
    for file in glob.glob(f'/global/cfs/cdirs/desi/users/arocher/Y1/2PCF_for_corr/Abcaus_small_boxes/z{zsim}/rppi/allcounts_AbacusSummit_small_c000_ph*.npy'):
        rp, wp = TwoPointCorrelationFunction.load(file)(pimax=pimax, return_sep=True)
        ax.plot(rp,rp*wp)
    axx[0].set_xscale('log')
    axx[0].set_xlabel(r'$r_p$ [Mpc/$h$]', fontsize=15)
    axx[0].set_ylabel(r'$r_p \cdot w_p$ [Mpc/$h$]$^2$', fontsize=15)
        
    for file in glob.glob(f'/global/cfs/cdirs/desi/users/arocher/Y1/2PCF_for_corr/Abcaus_small_boxes/z{zsim}/smu/allcounts_AbacusSummit_small_c000_ph*.npy'):
        s, xi = TwoPointCorrelationFunction.load(file)(ells=ells, return_sep=True)
        axx[1].plot(s,s*xi[0])
        axx[2].plot(s,s*xi[1])
        
    axx[1].set_xscale('log')
    axx[1].set_xlabel(r'$s$ [Mpc/$h$]', fontsize=15)
    axx[1].set_ylabel(r'$s \cdot \xi_0$ [Mpc/$h$]$^2$', fontsize=15)
    axx[2].set_xscale('log')
    axx[2].set_xlabel(r'$s$ [Mpc/$h$]', fontsize=15)
    axx[2].set_ylabel(r'$s \cdot \xi_2$ [Mpc/$h$]$^2$', fontsize=15)
    axx[0].grid()
    axx[1].grid()
    axx[2].grid()
    fig.tight_layout()


parser = argparse.ArgumentParser()

parser.add_argument('--dir_param_file', help='path to param file', type=str)
parser.add_argument('--phase', help='phase of abacus small boxes', type=int)
parser.add_argument('--zsim', help='simulation redshift', type=float)
parser.add_argument('--dir_to_save', help='directory to save the results', type=str, default='/global/cfs/cdirs/desi/users/arocher/Y1/2PCF_for_corr/Abcaus_small_boxes')
args = parser.parse_args()


param = yaml.load(open(args.dir_param_file), Loader=yaml.FullLoader)
ph = str(args.phase).zfill(3)
param['hcat']['Abacus']['z_simu'] = args.zsim
dir_to_save = args.dir_to_save

rppi = TwoPointCorrelationFunction.load('/global/cfs/cdirs/desi/survey/catalogs/edav1/xi/sv3/rppi/allcounts_ELG_NScomb_0.8_1.1_default_angular_bitwise_FKP_log_njack128_nran18_split20.npy')
smu = TwoPointCorrelationFunction.load('/global/cfs/cdirs/desi/survey/catalogs/edav1/xi/sv3/smu/allcounts_ELG_NScomb_0.8_1.1_default_angular_bitwise_FKP_log_njack128_nran18_split20.npy')

param['2PCF_settings']['edges_rppi'] = rppi.edges
param['2PCF_settings']['edges_smu'] = smu.edges

com_tr = np.vstack([np.array(np.meshgrid(param['tracers'],param['tracers'])).T.reshape(-1, len(param['tracers'])).flatten().reshape(len(param['tracers']),len(param['tracers']),2)[i,i:] for i in range(len(param['tracers']))])
str_comb = ['_'.join(np.unique(tt)) for tt in com_tr]

stats= ['rppi', 'smu'] if ('wp' in param['fit_param']['fit_type']) & ('xi' in param['fit_param']['fit_type']) else ['rppi'] if ('wp' in param['fit_param']['fit_type']) else ['smu']
path_to_save = os.path.join(dir_to_save, 'z{:.3f}'.format(args.zsim), '{}', 'allcounts_{{}}_AbacusSummit_small_c000_ph{}.npy'.format(ph))

if all([os.path.exists(path_to_save.format(stats[-1], tt)) for tt in str_comb]):
    print('ph {} already done for {} at z{}'.format(ph, str_comb, args.zsim), flush=True)
    exit()
else:
    param['hcat']['Abacus']['sim_name'] = 'AbacusSummit_small_c000_ph{}'.format(ph)

    HOD_obj = HOD(args=param, path_to_abacus_sim='/global/cfs/cdirs/desi/cosmosim/Abacus')
    cats = HOD_obj.make_mock_cat(verbose=False) 
    mock_cats = [cats[cats['TRACER'] == tr] for tr in HOD_obj.args['tracers']]

    from itertools import combinations_with_replacement
    cat_comb = list(combinations_with_replacement(mock_cats, 2))
    tr_comb = list(combinations_with_replacement(HOD_obj.args['tracers'], 2))

    for cats, tr in zip(cat_comb, tr_comb):
        str_tr = '_'.join(np.unique(tr))
        if HOD_obj.args['2PCF_settings']['rsd']:
            pos = apply_rsd(cats[0], HOD_obj.args['hcat']['z_simu'], HOD_obj.boxsize, HOD_obj.cosmo, los=HOD_obj.args['2PCF_settings']['los'], )
            pos1 = apply_rsd(cats[1], HOD_obj.args['hcat']['z_simu'], HOD_obj.boxsize, HOD_obj.cosmo, los=HOD_obj.args['2PCF_settings']['los'])
        else:
            pos = cats[0]['x']%HOD_obj.boxsize, cats[0]['y']%HOD_obj.boxsize, cats[0]['z']%HOD_obj.boxsize
            pos1 = cats[1]['x']%HOD_obj.boxsize, cats[1]['y']%HOD_obj.boxsize, cats[1]['z']%HOD_obj.boxsize
        

        for corr_type in stats:
            result = TwoPointCorrelationFunction(corr_type, HOD_obj.args['2PCF_settings'][f'edges_{corr_type}'], 
                                                    data_positions1=pos, data_positions2=pos1, engine='corrfunc', 
                                                    boxsize=HOD_obj.boxsize, los='z', nthreads=64)
            os.makedirs(os.path.dirname(path_to_save).format(corr_type), exist_ok=True)
            result.save(path_to_save.format(corr_type, str_tr))

        print('ph {} done'.format(ph), flush=True)
  