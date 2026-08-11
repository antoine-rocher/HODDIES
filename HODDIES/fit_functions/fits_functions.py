import numpy as np
import os
import glob

def genereate_training_points(nPoints, name_param, priors_lists, sampling_type='lhs', path_to_save_training_point=None, rand_seed=None):
    """
    Generate points in a LHS or Hammersley sample for a given priors

    Parameters
    ----------
    nPoints : float
        Number of points for the training sample
    name_param: list
        List of parameter names
    priors_lists: list
        List of prior ranges (min, max) for each parameter
    sampling_type: str, default: lhs
        Sampling type: 'lhs' or 'Hammersley' only
    path_to_save_training_point: str, default: None
        Directory to save the sample of points. If path_to_save_training_point is provided the sample will be save with name file points_{sampling_type}.txt as npz file with keys 'sample' and 'header'. 
    rand_seed: int, default: None
        Random seed for lhs sampling
    Returns
    -------
    sample_points : float array of shape (nPoints, len(priors))
        Sample of points according to the sampling type. 
    """
    from idaes.core.surrogate.pysmo.sampling import HammersleySampling, LatinHypercubeSampling, UniformSampling, HaltonSampling, CVTSampling

    reshape_priors = np.vstack(priors_lists).T.tolist()
    if sampling_type.lower() == 'lhs':
        res = LatinHypercubeSampling(reshape_priors, nPoints, rand_seed=rand_seed).sample_points()
    elif sampling_type.lower() == 'hammersley':
        res = HammersleySampling(reshape_priors, nPoints).sample_points()
    elif sampling_type.lower() == 'uniform':
        res = UniformSampling(reshape_priors, nPoints, rand_seed=rand_seed).sample_points()
    elif sampling_type.lower() == 'halton':
        res = HaltonSampling(reshape_priors, nPoints).sample_points()
    elif sampling_type.lower() == 'cvt':
        res = CVTSampling(reshape_priors, nPoints).sample_points()
    elif sampling_type.lower() == 'random':
        np.random.seed(rand_seed)
        res = np.random.uniform(reshape_priors[0], reshape_priors[1], size=(nPoints, len(reshape_priors[0])))
    else:
        raise ValueError(
            'Wrong sampling type. Available sampling types are "lhs", "hammersley", "uniform", "halton", "cvt", or "random".')
    
    sample_points = np.zeros((nPoints), dtype=[(name, dt) for name, dt in zip(name_param, ['float64']*len(name_param))])
    for i,v in enumerate(name_param):
        sample_points[v] = res.T[i] 
    if path_to_save_training_point is not None:
        os.makedirs(path_to_save_training_point, exist_ok=True)
        np.savez(os.path.join(path_to_save_training_point, f'list_HOD_parameters_{sampling_type}'),
                    sample=sample_points)
    
    return sample_points
                

def compute_chi2(model, data, inv_Cov2=None, sig=None):
    """
    Compute the chi-squared (χ²) statistic between a model and observed data.

    The function calculates χ² based on either the standard deviation of errors 
    (`sig`) or the inverse covariance matrix (`inv_Cov2`), depending on which is provided.

    Parameters:
    -----------
    model : array_like
        The predicted/model values.
    data : array_like
        The observed data values.
    inv_Cov2 : array_like, optional
        The inverse of the covariance matrix of the data. Used if `sig` is not provided.
    sig : array_like, optional
        The standard deviation (1σ errors) of the data. If provided, 
        chi-squared is computed as the sum of squared residuals normalized by the variance.

    Returns:
    --------
    chi2 : float
        The computed chi-squared value.

    Raises:
    -------
    ValueError
        If neither `sig` nor `inv_Cov2` is provided.
    """
    arr_diff = data - model
    if inv_Cov2 is not None:
        chi2 = np.matmul(arr_diff, np.matmul(inv_Cov2, arr_diff.T))
    elif sig is not None:
        chi2 = np.sum(arr_diff**2 / sig**2)
    else:
        raise ValueError('No error provided')
    return chi2


def multivariate_gelman_rubin(chains):
    """
    Compute the multivariate Gelman-Rubin convergence diagnostic (R̂) for MCMC chains.

    This diagnostic checks for convergence across multiple Markov Chain Monte Carlo (MCMC)
    simulations by comparing the between-chain and within-chain variances. Values close to 1
    indicate convergence.

    This is based on the method described in:
    Brooks, S. P., & Gelman, A. (1998). "General methods for monitoring convergence of iterative simulations."
    http://www.stat.columbia.edu/~gelman/research/published/brooksgelman2.pdf

    Parameters:
    -----------
    chains : array_like
        A list or array of MCMC chains with shape (nchains, nsteps, ndim), where:
            - nchains is the number of parallel chains,
            - nsteps is the number of samples per chain,
            - ndim is the number of parameters per sample.

    Returns:
    --------
    float
        The maximum eigenvalue of the matrix used in the multivariate R̂ computation.
        A value significantly greater than 1 indicates lack of convergence.

    Raises:
    -------
    AssertionError
        If the inversion of Wn1 does not yield an identity matrix within numerical precision.
    """
    nchains = len(chains)
    mean = np.asarray([np.mean(chain, axis=0) for chain in chains])
    variance = np.asarray([np.cov(chain.T, ddof=1) for chain in chains])
    nsteps = np.asarray([len(chain) for chain in chains])

    Wn1 = np.mean(variance, axis=0)
    Wn = np.mean(((nsteps - 1.) / nsteps)[:, None, None] * variance, axis=0)
    B = np.cov(mean.T, ddof=1)
    V = Wn + (nchains + 1.) / nchains * B

    invWn1 = np.linalg.inv(Wn1)
    assert np.absolute(Wn1.dot(invWn1) - np.eye(Wn1.shape[0])).max() < 1e-5

    eigen = np.linalg.eigvalsh(invWn1.dot(V))
    return eigen.max()


def values_in_boundaries(values, boundaries):
    """
    Check whether each value lies within any of the given boundary intervals.

    Parameters
    ----------
    values : array-like of float
        Values to test.

    boundaries : array-like of shape (n_intervals, 2)
        Each element is [low, high], defining an inclusive interval.
        Example:
            [[12.5, 13.5], [13, 14.5], [0.5, 1.5], [0.05, 1]]

    Returns
    -------
    np.ndarray of bool
        Boolean mask (len(values),) indicating whether each value lies
        inside at least one interval.

    Examples
    --------
    >>> vals = [0.8, 1.2, 13.1, 15]
    >>> bounds = [[12.5, 13.5], [13, 14.5], [0.5, 1.5], [0.05, 1]]
    >>> values_in_boundaries(vals, bounds)
    False
    """
    values = np.asarray(values)
    boundaries = np.asarray(boundaries, dtype=float)

    if boundaries.ndim != 2 or boundaries.shape[1] != 2:
        raise ValueError("boundaries must be an array of shape (n, 2) with [low, high] pairs")

    lows = boundaries[:, 0]   # shape (n_intervals, 1)
    highs = boundaries[:, 1] # shape (n_intervals, 1)
    # Vectorized interval membership test
    return  ((values >= lows) & (values <= highs)).all()


def func_stochopy(new_params, HOD_obj, fix_seed=10, verbose=False): 
    """
    Compute the chi-squared (χ²) statistic for a given set of parameters.

    Parameters
    ----------
    new_params : array-like
        New parameter values to evaluate.
    HOD_obj : object
        An instance of the HOD class containing methods for model evaluation.
    fix_seed : int, optional
        Random seed for reproducibility. Default is 10.
    verbose : bool, optional
        If True, print additional information during computation. Default is False.
    """
    
    HOD_obj.logger.info('New param: '+(', ').join([f'{nn}: {val}' for nn, val in zip(HOD_obj.name_params, new_params)]))
    new_params= np.array([new_params])
    if not values_in_boundaries(new_params, HOD_obj.priors):
        HOD_obj.logger.error('param outside priors {}'.format(new_params))
        return np.inf
    new_params.dtype = [(name, dt) for name, dt in zip(HOD_obj.name_params, ['float64']*len(HOD_obj.name_params))]
    HOD_obj.update_new_param(new_params, HOD_obj.name_params)
    cats = HOD_obj.make_mock_cat(fix_seed=fix_seed, verbose=verbose) 
    result = HOD_obj.compute_stats(cats, stat=HOD_obj._stats_to_fit, tracers=HOD_obj._tracer_to_fit, verbose=verbose)
    res = np.hstack([np.hstack([np.hstack(result[stat][comb_tr][-1])for stat in HOD_obj._stats_to_fit]) for comb_tr in HOD_obj._comb_tr_list])
    
    chi2 = compute_chi2(res, HOD_obj.data, HOD_obj.inv_cov2)
    HOD_obj.logger.info(f'chi2={chi2}')
    return chi2


def get_corr_small_boxes(param, tracers, **kwargs):

    from pycorr import TwoPointCorrelationFunction, utils

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


def load_desi_data(fit_param, tracers, **kwargs):

    from pycorr import TwoPointCorrelationFunction
    
    fit_param.update(kwargs)
    if ('ELG' in tracers) & ~('EDA' in fit_param['dir_data']) :
        tracers[tracers.index('ELG')] = 'ELG_LOPnotqso'
    
    com_tr = np.vstack([np.array(np.meshgrid(tracers,tracers)).T.reshape(-1, len(tracers)).flatten().reshape(len(tracers),len(tracers),2)[i,i:] for i in range(len(tracers))])
    print('Load data vector for', '_'.join(map(str, np.unique(com_tr))),  flush=True)
    
    # data = {'wp': {}, 'xi': {}}  if ('wp' in fit_param['fit_type']) & ('xi' in fit_param['fit_type']) else {'wp': {}}  if ('wp' in fit_param['fit_type']) else {'xi': {}}
    data = {}
    data['edges'] = {}
    stats= ['rppi', 'smu'] if ('wp' in fit_param['fit_type']) & ('xi' in fit_param['fit_type']) else ['rppi'] if ('wp' in fit_param['fit_type']) else ['smu']
    if fit_param['load_cov_jk'] is not None:
        cov_jk11 = []
    for tr in com_tr:
        str_tr = '_'.join(map(str, np.unique(tr)))
        file_name = f"allcounts_{{}}_{fit_param['region']}_{fit_param['zmin']}_{fit_param['zmax']}_{fit_param['weights_type']}_{fit_param['bin_type']}_njack{fit_param['njack']}_nran{fit_param['nran']}_split20.npy"
        file_dir = os.path.join(fit_param['dir_data'], '{}', file_name)

        data[f'{tr[0][:3]}_{tr[1][:3]}'] = {}   
        cov_jk1 = []
        for corr in stats:
            try:
                result =  TwoPointCorrelationFunction.load(file_dir.format(corr, str_tr))
            except FileNotFoundError as e:
                str_tr = '_'.join(map(str, np.unique(tr)[::-1]))
                result =  TwoPointCorrelationFunction.load(file_dir.format(corr, str_tr))
            print(f'Load {corr} measurements at z{fit_param["zmin"]}-{fit_param["zmax"]} for {str_tr}...', flush=True)

            if corr == 'rppi':
                result= result[::2][4:-3]
                data[f'{tr[0][:3]}_{tr[1][:3]}']['wp'] = result(return_sep=True, return_std=True, pimax=fit_param['pimax'])
                data['edges']['wp'] = result.edges
                if fit_param['load_cov_jk'] is not None:
                    cov_jk1 += [[result.realization(i)(pimax=fit_param['pimax']) for i in range(result.nrealizations)]]

            elif corr == 'smu':
                result= result[16:-6]
                data[f'{tr[0][:3]}_{tr[1][:3]}']['xi'] = result(return_sep=True, return_std=True, ells=fit_param['multipole_index'])
                data['edges']['xi'] = result.edges
                if fit_param['load_cov_jk'] is not None:
                    cov_jk1 += [np.vstack([np.hstack(result.realization(i)(ells=fit_param['multipole_index'])) for i in range(result.nrealizations)])]
        if fit_param['load_cov_jk'] is not None: cov_jk11 += [np.hstack(cov_jk1)]
        
    if fit_param['load_cov_jk'] is not None:
        data['cov_jk'] = ((result.nrealizations-1) * np.cov(np.hstack(cov_jk11), rowvar=False, ddof=0))
    if 'ELG_LOPnotqso' in tracers: tracers[tracers.index('ELG_LOPnotqso')] = 'ELG'
    return data



def get_hartlap_factor(nmocks, ndata):
    """
    Compute the Hartlap factor for covariance matrix correction.

    The Hartlap factor is used to correct the inverse of a covariance matrix 
    estimated from a finite number of mock realizations. It accounts for the 
    bias introduced when inverting a sample covariance matrix.

    Parameters
    ----------
    nmocks : int
        Number of mock realizations used to estimate the covariance matrix.
    ndata : int
        Number of data points (dimensions) in the data vector.

    Returns
    -------
    float
        The Hartlap factor, which is applied to the inverse covariance matrix.
        It is given by: (nmocks - ndata - 2) / (nmocks - 1)
        If nmocks <= ndata + 2, a warning is issued and the factor is set to 1.0.
    """
    if nmocks <= ndata + 2:
        print("Warning: Number of mocks is too small compared to number of data points. Hartlap factor set to 1.0.")
        return 1.0
    return (nmocks - ndata - 2) / (nmocks - 1)

