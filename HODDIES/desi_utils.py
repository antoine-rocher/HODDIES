import glob
from pycorr import TwoPointCorrelationFunction, utils
import os 
import numpy as np


def get_corr_small_boxes(zsim, corr, parent_dir='/global/cfs/cdirs/desi/users/arocher/HODDIES_results/Abcaus_small_boxes_corr', pimax=40, ells=[0,2], return_cov=False):
    wp, xi = None, None
    files = os.path.join(parent_dir, 'z{zsim:.3f}/{{}}/allcounts_*.npy'.format(zsim=zsim))
    num_sim = len(glob.glob(files.format(corr[0])))
    if 'rppi' in corr:
        print(f'Loading rppi correlation function for z={zsim}...')
        wp = [TwoPointCorrelationFunction.load(file)[::2][4:-3](pimax=pimax) for file in glob.glob(files.format('rppi'))]

    if 'smu' in corr:
        print(f'Loading smu correlation function for z={zsim}...')
        xi = [np.hstack(TwoPointCorrelationFunction.load(file)[16:-6](ells=ells)) for file in glob.glob(files.format('smu'))]
        
    if wp is None:
        cov = np.cov(xi, rowvar=False, ddof=1) 
        corr = utils.cov_to_corrcoef(cov)
    elif xi is None:
        cov = np.cov(wp, rowvar=False, ddof=1)  
        corr = utils.cov_to_corrcoef(cov)
    else:
        cov = np.cov(np.hstack((wp,xi)), rowvar=False, ddof=1)
        corr = utils.cov_to_corrcoef(cov)
    if return_cov:
        return cov, num_sim
    return corr


def get_corr_small_boxes_tr(param, tracers, **kwargs):

    param.update(kwargs)
    com_tr = np.vstack([np.array(np.meshgrid(tracers,tracers)).T.reshape(-1, len(tracers)).flatten().reshape(len(tracers),len(tracers),2)[i,i:] for i in range(len(tracers))])
    str_tr_all = '_'.join(map(str, np.unique(com_tr)))
    corr_fn = os.path.join(param['corr_dir'], f'z{param["z_simu"]:.3f}', f'Mcorr_small_box_z{param["z_simu"]}_{param["fit_type"]}_{str_tr_all}.txt')

    if os.path.exists(corr_fn):
        print(f'Load correlation matrix for {str_tr_all} at z{param["z_simu"]} ...', flush=True)
        corr = np.loadtxt(corr_fn)
        return corr
    print(f'Load correlation matrix for {str_tr_all}...', flush=True)
    corr_type= ['rppi', 'smu'] if ('wp' in param['fit_type']) & ('xi' in param['fit_type']) else ['rppi'] if ('wp' in param['fit_type']) else ['smu']
    res = []
    for tr in com_tr:
        tr = np.unique(tr)
        str_tr = '_'.join(map(str, np.unique(tr)))
        file_name = f'allcounts_{str_tr}_AbacusSummit_small_c000_ph*.npy'
        file_dir = os.path.join(param['corr_dir'], f'z{param["z_simu"]:.3f}', '{}', file_name)

        res_tr = []
        for corr_t in corr_type:
            if os.path.exists(os.path.join(file_dir.format(corr_t, str_tr))):
                pass
            else:
                str_tr = '_'.join(map(str, np.unique(tr)[::-1]))
            fns = glob.glob(os.path.join(file_dir.format(corr_t, str_tr)))
            if len(fns) == 0:
                raise FileNotFoundError(f'No {corr_t} measurements at z{param["z_simu"]} for {str_tr}...')
            print(f'Load {corr_t} measurements at z{param["z_simu"]} for {str_tr}...', flush=True)
            if corr_t == 'rppi':
                res_tr += [[TwoPointCorrelationFunction.load(file)[::2][4:-3](pimax=param['pimax']) for file in fns]]
            else:
                res_tr += [[np.hstack(TwoPointCorrelationFunction.load(file)[16:-6](ells=param['multipole_index'])) for file in fns]]
        # print(len(glob.glob(f'/global/cfs/cdirs/desi/users/arocher/Y1/2PCF_for_corr/Abcaus_small_boxes/z{param["z_simu"]:.3f}/smu/allcounts_{param["z_simu"]}_AbacusSummit_small_c000_ph*.npy')))
        # return 0
        res += [np.hstack(res_tr)]        

    corr = utils.cov_to_corrcoef(np.cov(np.hstack(res), rowvar=False, ddof=0))
    np.savetxt(os.path.join(param['corr_dir'], f'z{param["z_simu"]:.3f}', f'Mcorr_small_box_z{param["z_simu"]}_{param["fit_type"]}_{str_tr_all}.txt'), corr)

    return corr