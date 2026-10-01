"""
Échantillonnage du modèle HOD complet (pas l'émulateur) avec pocoMC,
parallélisé en MPI, chaque tâche exploitant le multithreading numba.

ARCHITECTURE
------------
    N_MPI tâches  x  N_THREADS threads numba  =  cœurs totaux

Chaque rang MPI évalue une vraisemblance à la fois, en utilisant ses
N_THREADS cœurs via numba.prange à l'intérieur de compute_N et
_compute_sat_from_abacus_part. Le pool MPI distribue les points du sampler
sur les rangs.

Le cloisonnement est essentiel : si N_THREADS n'est pas fixé explicitement,
numba prend tous les cœurs visibles dans chaque rang et les rangs se
disputent les mêmes cœurs, ce qui est plus lent que le mode série.

LE BRUIT STOCHASTIQUE
---------------------
Le peuplement HOD est aléatoire, donc log L(theta) est bruité. Sans
traitement, les chaînes diffusent dans un plateau de bruit et ne convergent
pas. Deux stratégies, contrôlées par `n_realizations` :

    n_realizations = 1  -> graine déterministe en theta. La vraisemblance
                           devient une fonction déterministe, mais rugueuse.
    n_realizations > 1  -> moyenne sur plusieurs peuplements. Le bruit décroît
                           en 1/sqrt(n), au prix d'un coût proportionnel.

Commencez à 1 avec graine fixée ; passez à 5-10 si la chaîne peine à se
resserrer autour du maximum.
"""

import os
import sys

# --- doit précéder tout import de numba / numpy ---------------------------
N_THREADS = int(os.environ.get('HOD_NUM_THREADS', '8'))
os.environ['NUMBA_NUM_THREADS'] = str(N_THREADS)
os.environ['OMP_NUM_THREADS']   = str(N_THREADS)
os.environ['MKL_NUM_THREADS']   = str(N_THREADS)

import numpy as np
import numba
from schwimmbad import MPIPool
import pocomc as pc
from scipy.stats import uniform
from HODDIES.fit_functions import values_in_boundaries, compute_chi2


numba.set_num_threads(N_THREADS)


# ==========================================================================
# Vraisemblance
# ==========================================================================
class HODLikelihood():
    """
    Objet appelable, picklable, qui garde l'état lourd (halos, sous-échantillon
    de particules) en mémoire dans chaque rang plutôt que de le transmettre à
    chaque appel.
    """

    def __init__(self, hod_obj, base_seed=42):
        self.hod            = hod_obj
        self.base_seed      = base_seed


    def __call__(self, new_params):
        new_params= np.asarray(new_params)

        self.hod.logger.info('New param: '+(', ').join([f'{nn}: {val}' for nn, val in zip(self.hod.name_params, new_params)]))
        new_params= np.array([new_params])
        if not values_in_boundaries(new_params, self.hod.priors):
            self.hod.logger.error('param outside priors {}'.format(new_params))
            return np.inf
        new_params.dtype = [(name, dt) for name, dt in zip(self.hod.name_params, ['float64']*len(self.hod.name_params))]
        self.hod.update_new_param(new_params, self.hod.name_params)
        cats = self.hod.make_mock_cat(fix_seed=self.base_seed, verbose=False) 
        result = self.hod.compute_stats(cats, stat=self.hod._stats_to_fit, tracers=self.hod._tracer_to_fit, verbose=False)
        res = np.hstack([np.hstack([np.hstack(result[stat][comb_tr][-1])for stat in self.hod._stats_to_fit]) for comb_tr in self.hod._comb_tr_list])
        
        chi2 = compute_chi2(res, self.hod.data, self.hod.inv_cov2)
        
        return float(-0.5 * chi2)


# ==========================================================================
# Prior
# ==========================================================================
def build_prior(boundaries):
    b = np.asarray(boundaries, dtype=float)
    return pc.Prior([uniform(lo, hi - lo) for lo, hi in b])


def HOD_LH(new_params, HOD_obj, fix_seed=10, verbose=False): 
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
    return -0.5*chi2

# ==========================================================================
# Driver
# ==========================================================================
def main():
    from mpi4py import MPI
    import argparse
    import yaml
    from HODDIES.sim_loader import AbacusSummitSim
    from HODDIES import HOD
    from HODDIES.fit_functions import get_hartlap_factor
    import glob
    from pycorr import utils
    comm = MPI.COMM_WORLD
    import mpytools as mpy
    
    mpy.CurrentMPIComm.enable(MPI.COMM_SELF)  # bascule TOUT mpytools sur COMM_SELF
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--param_file', help='parameter file', type=str)
    # parser.add_argument('--seed_training', help='seed to use for generating training points', type=int, default=1234567489)
    # parser.add_argument('--run_test_set', help='run test set instead of training set', action='store_true', default=False)
    # parser.add_argument('--overwrite', help='overwrite the training sample if it already exist', action='store_true', default=False)
    # args = parser.parse_args()
    param_file = '/global/homes/a/arocher/Code/postdoc/HODDIES_v2/HODDIES/HODDIES/fit_functions/scripts/example_HOD_fit_parameter_mcmc.yaml'
    param = yaml.safe_load(open(param_file, 'r'))
    param['nthreads'] = N_THREADS

    sim_abacus=AbacusSummitSim(**param['hcat'])

    HOD_obj = HOD(base_catalog=sim_abacus, setup_logger=False, **param)

    HOD_obj._initialize_fit_params()


    # --- chargement de l'état lourd, identique dans tous les rangs --------
    # Chaque rang charge sa propre copie : les halos ne se transmettent pas
    # par pickle à chaque appel de vraisemblance.
    files = glob.glob('/global/cfs/cdirs/desi/users/arocher/HODDIES_results/test/AbacusSummit_base_c000_ph000/z0.500/mock_data/*')
    files.sort()
    ff = files[0]   
    mcov= np.loadtxt('/global/cfs/cdirs/desi/users/arocher/HODDIES_results/Abcaus_small_boxes_corr/z0.500/Mcorr_small_box_z0.500_wp_xi_ells_LRG_SHOD_6p.txt')
    corr = utils.cov_to_corrcoef(mcov)
    num_sim = 1786 # number of Abacus small boxes used to compute the covariance matrix
    mcov = mcov/64 # to rescale for the volume of the Abacus small boxes (1/64 of the base volume)

    mock_data_dict = np.load(ff, allow_pickle=True)[()]
    data_vec = np.hstack([mock_data_dict['wp']['LRG_LRG'][-1][0], np.hstack(mock_data_dict['xi_ells']['LRG_LRG'][-1].reshape(20,2,26)[0])])
    err_data_vec = np.hstack([mock_data_dict['wp']['LRG_LRG'][-1].std(axis=0), np.hstack(mock_data_dict['xi_ells']['LRG_LRG'][-1].reshape(20,2,26).std(axis=0))])


    hartlap_fac = get_hartlap_factor(num_sim, len(mcov))
    corr = utils.cov_to_corrcoef(mcov)
    sig_data = np.sqrt(np.diag(mcov) + err_data_vec**2)

    Mcov_data = corr*sig_data*sig_data[:,None]
    HOD_obj.initialize_fit(data_vec=data_vec, err=Mcov_data, hartlap_factor=hartlap_fac, add_poisson_noise=False, verbose=True)
    
    
    loglike = HODLikelihood(HOD_obj, base_seed=42)
    print(f'cat csize     : {HOD_obj.hcat.csize}', flush=True)
    print(f'cat size     : {HOD_obj.hcat.size}', flush=True)

    with MPIPool() as pool:
        if not pool.is_master():
            pool.wait()
            sys.exit(0)

        print(f'cat logM     : {HOD_obj.hcat["log10_Mh"][:5]}', flush=True)
        print(f'rangs MPI      : {comm.Get_size()}', flush=True)
        print(f'threads/rang   : {N_THREADS}', flush=True)
        print(f'cœurs totaux   : {comm.Get_size() * N_THREADS}', flush=True)
        print(f'dimensions     : {len(HOD_obj.name_params)}', flush=True)

        sampler = pc.Sampler(
            prior=build_prior(HOD_obj.priors),
            likelihood=loglike,
            pool=pool,
            n_effective=512,
            n_active=256,
            random_state=42,
            output_dir='/global/cfs/cdirs/desi/users/arocher/HODDIES_results/test/mcmc/pocomc_out',      # reprise possible après un timeout
        )

        # sampler.run(n_total=5000, n_evidence=2500, progress=True)
        sampler.run(n_total=10, n_evidence=5, progress=True, save_every=5)

        samples, weights, logl, logp = sampler.posterior()
        logz, logz_err = sampler.evidence()

        print(f'log Z = {logz:.3f} +/- {logz_err:.3f}')
        np.save('/global/cfs/cdirs/desi/users/arocher/HODDIES_results/test/mcmc/hod_samples.npy', np.asarray(samples))
        np.save('/global/cfs/cdirs/desi/users/arocher/HODDIES_results/test/mcmc/hod_weights.npy', np.asarray(weights))


if __name__ == '__main__':
    main()
