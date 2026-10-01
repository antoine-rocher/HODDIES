from HODDIES.fit_functions.FCNN_utils import make_training_dataset, train_model
from HODDIES.desi_utils import get_corr_small_boxes
import emcee 
import numpy as np
from pycorr import utils
import glob
from HODDIES.fit_functions.fits_functions import get_hartlap_factor, likelihood, likelihood_vec
from chainconsumer import Chain, ChainConsumer, Truth
from HODDIES.fit_functions.plotting_emulator_func import plot_bf_results
import sys

param_labels = {
    "Ac": r"$A_c$",
    "As": r"$A_s$",
    "M_0": r"$M_0$",
    "M_1": r"$M_1$",
    "Q": r"$Q$",
    "alpha": r"$\alpha$",
    # --- assembly bias parameters ---
    "ab_c_cen": r"$A_{B,\,c}^{\mathrm{cen}}$",
    "ab_c_sat": r"$A_{B,\,c}^{\mathrm{sat}}$",
    "ab_env_cen": r"$A_{B,\,\mathrm{env}}^{\mathrm{cen}}$",
    "ab_env_sat": r"$A_{B,\,\mathrm{env}}^{\mathrm{sat}}$",
    # --- other model parameters ---
    "f_sigv": r"$f_{\sigma_v}$",
    "gamma": r"$\gamma$",
    "log_Mcent": r"$\log M_{\mathrm{cent}}$",
    "pmax": r"$p_{\max}$",
    "sigma_M": r"$\sigma_M$",
    "exp_frac": r"$f_{\exp}$",
    "exp_scale": r"$s_{\exp}$",
    "nfw_rescale": r"$\lambda_{\mathrm{NFW}}$",
    "v_infall": r"$v_{\mathrm{infall}}$",
    "v_smear": r"$v_{\mathrm{smear}}$"
}


dataset_path = '/global/cfs/cdirs/desi/users/arocher/HODDIES_results/test/AbacusSummit_base_c000_ph000/z0.500/training_set/LRG_SHOD_6p/Hammersley/'
testset_path='/global/cfs/cdirs/desi/users/arocher/HODDIES_results/test/AbacusSummit_base_c000_ph000/z0.500/test_set/LRG_SHOD_6p/lhs/'
path_to_model='/global/homes/a/arocher/Code/postdoc/HODDIES_v2/HODDIES/HODDIES/fit_functions/saved_model/test_model_LRG_SHOD_6p_z0.5_small_wp+xi_ells.pth'
# path_to_model=None
# testset_path=None
num_testset=200
train_Dataset, train_loader, val_loader = make_training_dataset(dataset_path, stats=['wp', 'xi_ells'], log_transform=['wp'], seed=421, batch_size=256)
# model = train_model(train_loader=train_loader, min_epochs=100, val_l  oader=val_loader, use_std=False, Activation_fn="SiLU", path_to_model=path_to_model)
file_best_fit_train = '/global/homes/a/arocher/Code/postdoc/HOD/Cosmological_emulator_ELGs/best_train_model.npy'

model = train_model(train_loader=train_loader, min_epochs=100, val_loader=val_loader, file_best_fit_train=file_best_fit_train, use_std=False,
                    Activation_fn="SiLU", path_to_model=path_to_model)

# mcov, num_sim = get_corr_small_boxes(zsim=0.5, corr=['rppi', 'smu'], pimax=40, ells=[0,2], return_cov=True)
mcov= np.loadtxt('/global/cfs/cdirs/desi/users/arocher/HODDIES_results/Abcaus_small_boxes_corr/z0.500/Mcov_small_box_z0.500_wp_xi_ells_LRG_SHOD_6p.txt')
corr = utils.cov_to_corrcoef(mcov)
num_sim = 1786 # number of Abacus small boxes used to compute the covariance matrix
mcov = mcov/64 # to rescale for the volume of the Abacus small boxes (1/64 of the base volume)

std_data = np.sqrt(np.diag(mcov))

nb_start = int(sys.argv[1])
nb = int(sys.argv[2])

files = glob.glob('/global/cfs/cdirs/desi/users/arocher/HODDIES_results/test/AbacusSummit_base_c000_ph000/z0.500/mock_data/*')
files.sort()

import torch
print(f"Number of CUDA devices: {torch.cuda.device_count()}")

for ff in files[nb_start:nb_start+nb]:
    print(f"Running MCMC for file: {ff}")
    mock_data_dict = np.load(ff, allow_pickle=True)[()]
    x_truth = mock_data_dict['hod_fit_param'].tolist()
    data_vec = np.hstack([mock_data_dict['wp']['LRG_LRG'][-1][0], np.hstack(mock_data_dict['xi_ells']['LRG_LRG'][-1].reshape(20,2,26)[0])])
    err_data_vec = np.hstack([mock_data_dict['wp']['LRG_LRG'][-1].std(axis=0), np.hstack(mock_data_dict['xi_ells']['LRG_LRG'][-1].reshape(20,2,26).std(axis=0))])


    hartlap_fac = get_hartlap_factor(num_sim, len(mcov))
    corr = utils.cov_to_corrcoef(mcov)
    sig_data = np.sqrt(np.diag(mcov) + err_data_vec**2)

    sign, logdet_corr_data = np.linalg.slogdet(corr)

    Mcov_data = corr*sig_data*sig_data[:,None]

    inv_corr_data = np.linalg.inv(corr)

    n_dim = len(x_truth)
    n_chain = 20

    priors = np.vstack([train_Dataset.X_training.min(axis=0), np.round(train_Dataset.X_training.max(axis=0), 2)]).T
    pos = np.array([np.random.uniform(l, h, size=n_chain) for l, h in priors]).T



    if model.device.type == 'cpu':
        print("Running MCMC on CPU...")
        sampler = emcee.EnsembleSampler(n_chain, n_dim, likelihood, args=(model, train_Dataset, data_vec, priors, None, inv_corr_data, sig_data, logdet_corr_data))
    else:
        print("Running MCMC on GPU...")
        # On GPU
        sampler = emcee.EnsembleSampler(
            n_chain, n_dim, likelihood_vec,
            args=(model, train_Dataset, data_vec, priors, None, inv_corr_data, sig_data, logdet_corr_data),
            vectorize=True)
        
    n_point = 10000

    sampler.run_mcmc(pos, n_point, progress=True)
    save_path = ff.replace('mock_data', 'mcmc_results')
    np.save(save_path, sampler)
    save_plots = '/global/cfs/cdirs/desi/users/arocher/HODDIES_results/test/AbacusSummit_base_c000_ph000/z0.500/mcmc_results/plots/'

    print(f"Finished MCMC sampler for file: {ff}. Results saved to {save_path}", flush=True)

    chain_name = 'AbacusSummit_base_c000_ph' + ff.split('ph')[-1][:-4]
    name_params = [param_labels[par[:-4]] for par in train_Dataset.name_arr]
    chain = Chain.from_emcee(sampler, name_params, chain_name, discard=1000, thin=2, color="indigo", extents=dict(zip(name_params, priors)), walkers=20)


    consumer = ChainConsumer().add_chain(chain)
    consumer.plotter.config.extents = dict(zip(name_params, priors))
    consumer.add_truth(Truth(location=dict(zip(name_params, x_truth))))



    fig = consumer.plotter.plot_walks(filename=f'{save_plots}walkers_{chain_name}.png')
    fig = consumer.plotter.plot(filename=f'{save_plots }chains_{chain_name}.png')
    
    summary = consumer.analysis.get_summary()
    bound = summary[chain_name]
    # print(bound.center)                 # median
    # print(bound.lower, bound.upper)     # 16th/84th by default
    x_best_fit = [bound[key].center for key in name_params]

    plot_bf_results(train_Dataset, model, x_best_fit, data_vec, err_data_vec,x_truth=x_truth, save_fn=f'{save_plots }bf_plot_{chain_name}.png')
    print(f"Finished MCMC for file: {ff}. Results saved to {save_path} and plots saved to {save_plots}", flush=True)