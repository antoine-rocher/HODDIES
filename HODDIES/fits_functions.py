import numpy as np
from idaes.core.surrogate.pysmo.sampling import HammersleySampling, LatinHypercubeSampling
import os


def genereate_training_points(nPoints, priors, sampling_type='lhs', path_to_save_training_point=None, rand_seed=None):
    """
        --- Generate points in a LHS or Hammersley sample for a given priors

            Parameters
            ----------
            nPoints : float
                Number of points for the training sample
            priors: dict
                Dictionary of the priors range (min, max) for each parameter
            sampling_type: str, default: lhs
                Sampling type: 'lhs' or 'Hammersley' only
            path_to_save_training_point: bool, default: False
                Directory to save the sample of points. If path_to_save_training_point is provided the sample will be save with name file points_{sampling_type}.txt as npz file with keys 'sample' and 'header'. 
            rand_seed: int, default: None
                Random seed for lhs sampling
            Returns
            -------
            sample_points : float array of shape (nPoints, len(priors))
                Sample of points according to the sampling type. 
    """
    
    print(f"Creating {sampling_type} sample...", flush=True)
    priors_lists = np.vstack([list(priors[tr].values()) for tr in priors.keys()]).T.tolist() 
    
    if sampling_type == 'lhs':
        res = LatinHypercubeSampling(priors_lists, nPoints, rand_seed=rand_seed).sample_points()
    elif 'Hammersley' in sampling_type:
        res = HammersleySampling(priors_lists, nPoints).sample_points()
    else:
        raise ValueError(
            'Wrong sampling type only "lhs" or "Hammersley" allowed')
    
    name_param = [f'{var}_{tr}' for tr in priors.keys() for var in priors[tr].keys()]
    sample_points = np.zeros((nPoints), dtype=[(name, dt) for name, dt in zip(name_param, ['float64']*len(name_param))])
    for i,v in enumerate(name_param):
        sample_points[v] = res.T[i] 
    if path_to_save_training_point is not None:
        np.savez(os.path.join("path_to_training_point", f'point_{sampling_type}'),
                    sample=sample_points, header=str(priors))
    
    return sample_points
                

def compute_chi2(model, data, inv_Cov2=None, sig=None):
    """
    --- Compute chi2
    """
    arr_diff = data - model
    if sig is not None:
        chi2 = np.sum(arr_diff**2/sig**2)
    elif inv_Cov2 is not None:
        chi2 = np.matmul(arr_diff, np.matmul(inv_Cov2, arr_diff.T))
    else:
        raise ValueError('No error provided')
    return chi2


def multivariate_gelman_rubin(chains):
    """
    Arnaud de Mattia code's
    http://www.stat.columbia.edu/~gelman/research/published/brooksgelman2.pdf
    dim 0: nchains
    dim 1: nsteps
    dim 2: ndim
    """
    nchains = len(chains)
    mean = np.asarray([np.mean(chain, axis=0) for chain in chains])
    variance = np.asarray([np.cov(chain.T, ddof=1)
                            for chain in chains])
    nsteps = np.asarray([len(chain) for chain in chains])
    Wn1 = np.mean(variance, axis=0)
    Wn = np.mean(((nsteps-1.)/nsteps)[:, None, None]*variance, axis=0)
    B = np.cov(mean.T, ddof=1)
    V = Wn + (nchains+1.)/nchains*B
    invWn1 = np.linalg.inv(Wn1)
    assert np.absolute(
        Wn1.dot(invWn1)-np.eye(Wn1.shape[0])).max() < 1e-5
    eigen = np.linalg.eigvalsh(invWn1.dot(V))
    return eigen.max()










