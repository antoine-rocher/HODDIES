import numpy as np
from numba import njit, numba
import os
import scipy
from scipy.special import gammaln  # log(n!)
import math 

def apply_rsd(cat, z, boxsize, cosmo, H_0=100, los='z', vsmear=0):

    """
    Apply redshift-space distortions (RSD) to a galaxy catalog.

    Parameters
    ----------
    cat : dict
        Dictionary containing particle positions and velocities. Keys should include 'x', 'y', 'z', 'vx', 'vy', 'vz'.
    z : float
        Redshift at which to apply the distortions.
    boxsize : float
        Size of the simulation box in Mpc/h.
    cosmo : object
        Cosmology object from cosmoprimo.
    H_0 : float, optional
        Hubble constant in km/s/Mpc. Default is 100.
    los : {'x', 'y', 'z'}, optional
        Line-of-sight axis. Default is 'z'.
    vsmear : float, array-like, optional
        Add redshift errors in km/s. If vsmear is a number create a normal distribution $\mathcal{N} (0,v_\mathrm{smear)}$. Default is 0.

    Returns
    -------
    pos_rsd : list of ndarray
        List of arrays containing RSD-applied positions for x, y, and z.
    """

    rsd_factor = 1 / (1 / (1 + z) * H_0 * cosmo.efunc(z))   
    pos_rsd = [cat[p] % boxsize if p !=los else (cat[p] + (cat['v'+p] + vsmear)*rsd_factor) %boxsize for p in 'xyz']    
    return pos_rsd
    
    
# def compute_2PCF(pos1, edges, boxsize, ells=(0, 2), los='z', nthreads=32, R1R2=None, pos2=None, mpicomm=None):

#     """
#     Compute the 2-point correlation function multipoles for a periodic box.

#     Parameters
#     ----------
#     pos1 : array-like
#         Positions of sample 1 (e.g., galaxies or halos).
#     edges : list of arrays
#         Bin edges in separation (s, mu).
#     boxsize : float
#         Size of the simulation box.
#     ells : tuple of int, optional
#         Multipoles to project onto. Default is (0, 2).
#     los : {'x', 'y', 'z'}, optional
#         Line-of-sight direction. Default is 'z'.
#     nthreads : int, optional
#         Number of threads for parallel computation. Default is 32.
#     R1R2 : array-like, optional
#         Precomputed RR counts for normalization. Default is None.
#     pos2 : array-like, optional
#         Positions of sample 2 (for cross-correlations). Default is None.
#     mpicomm : object, optional
#         MPI communicator. Default is None.

#     Returns
#     -------
#     s, (multipoles) : tuple(array, ndarray)
#         Average separation for each s bin and computed multipoles of the 2PCF.
#     """    

#     result = TwoPointCorrelationFunction('smu', edges, 
#                                             data_positions1=pos1, data_positions2=pos2, engine='corrfunc', 
#                                             boxsize=boxsize, los=los, nthreads=nthreads, R1R2=R1R2, mpicomm=mpicomm)
    
#     return project_to_multipoles(result, ells=ells)

    
# def compute_wp(pos1, edges, boxsize, pimax=40, los='z', nthreads=32, R1R2=None, pos2=None, mpicomm=None):
#     """
#     Compute the projected correlation function w_p(r_p).

#     Parameters
#     ----------
#     pos1 : array-like
#         Positions of sample 1 (e.g., galaxies or halos).
#     edges : list of arrays
#         Bin edges for projected separation (r_p, pi).
#     boxsize : float
#         Size of the simulation box.
#     pimax : float, optional
#         Maximum line-of-sight separation for integration.
#     los : {'x', 'y', 'z'}, optional
#         Line-of-sight direction. Default is 'z'.
#     nthreads : int, optional
#         Number of threads for parallel computation. Default is 32.
#     R1R2 : array-like, optional
#         Precomputed RR counts for normalization. Default is None.
#     pos2 : array-like, optional
#         Positions of sample 2 for cross-correlations. Default is None.
#     mpicomm : object, optional
#         MPI communicator. Default is None.

#     Returns
#     -------
#     rp, wp : tuple(array, array)
#         Seperation and projected correlation function.
#     """

#     result = TwoPointCorrelationFunction('rppi', edges, 
#                                          data_positions1=pos1, data_positions2=pos2, engine='corrfunc', 
#                                          boxsize=boxsize, los=los, nthreads=nthreads, R1R2=R1R2, mpicomm=mpicomm)
    
#     return project_to_wp(result, pimax=pimax)


# def compute_delta_sigma(
#     pos_lens,   
#     pos_particles,
#     rbins,
#     boxsize,
#     rho_m,
#     los='z',
#     pimax=30,
#     nthreads=32,
# ):
#     """
#     Compute excess surface density ΔΣ(R) in units of 1e12[Msun/h / (Mpc/h)^2] 
#     using rp–π pair counts and fast vectorized Gauss–Legendre integration.

#     Parameters
#     ----------
#     pos_lens : (N_lens, 3) array
#         Lens positions.
#     pos_particles : (N_part, 3) array
#         Particle positions.
#     rbins : array
#         Projected-radius bin edges.
#     boxsize : float
#         Periodic-box size [Mpc/h].
#     rho_m : float
#         Mean matter density [Msun/h / (Mpc/h)^3].
#     pimax : float
#         Maximum LOS half-depth for Σ(R).
#     los : str
#         LOS axis for Corrfunc 'x', 'y' or 'z'. Default 'z'.
#     nthreads : int
#         number of threads to use for Corrfunc.

#     Returns
#     -------
#     delta_sigma : array
#         ΔΣ(R) in units of 1e12[Msun/h / (Mpc/h)^2].
#     """

#     # --- 1) rp–pi bins
#     rpbins = np.geomspace(0.001, rbins.max(), 100)
#     pibins = np.linspace(-pimax, pimax, 2 * pimax + 1)

#     # --- 2) Measure ξ(rp,pi)
#     rp, wp = TwoPointCorrelationFunction(
#         "rppi",
#         edges=[rpbins, pibins],
#         data_positions1=pos_lens,
#         data_positions2=pos_particles,
#         boxsize=boxsize,
#         los=los,
#         nthreads=nthreads
#     )(return_sep=True, pimax=pimax)

#     # --- 4) Sigma(R) = rho_m * w_p(R)
#     Sigma = rho_m * wp

#     # --- 5) Compute Σ(<R)
#     spline_S = interp1d(rp, Sigma, kind="cubic",
#                         bounds_error=False, fill_value="extrapolate")

#     def integrand(r):
#         return r * spline_S(r)

#     Sigma_mean = np.array([2.0 / R**2 * quad(integrand, 0, R, limit=200)[0] for R in rp])

#     # --- 6) Excess surface density
#     DeltaSigma = (Sigma_mean - Sigma) /1e12 
#     spline_Dsigma = interp1d(rp, DeltaSigma) 
#     rp_cent = 0.5 * (rbins[1:] + rbins[:-1])

#     return rp_cent, spline_Dsigma(rp_cent)




TAB_GAMMA_LN = scipy.special.gammaln(np.arange(1000001, dtype=np.float64))

@njit(fastmath=True)
def log_factorial(x):
    return TAB_GAMMA_LN[x + 1]


@njit(fastmath=True)
def cmp_logZ(lam, nu, max_x=1000):
    x = np.arange(max_x + 1)
    log_terms = x * np.log(lam) - nu * log_factorial(x)
    m = np.max(log_terms)          # log-sum-exp trick
    return m + np.log(np.sum(np.exp(log_terms - m)))

@njit(fastmath=True)
def cmp_pmf_vec(x, lam, nu, logZ=None, max_x=1000):
    x = np.asarray(x)
    if logZ is None:
        logZ = cmp_logZ(lam, nu, max_x)

    log_p = x * np.log(lam) - nu * log_factorial(x) - logZ
    return np.exp(log_p)

@njit(fastmath=True, cache=True)
def get_max_x(lam):
    if lam < 15:
        return 50
    elif lam < 100:
        return 200
    elif lam < 700:
        return 1000
    elif lam < 4000:
        return 5000
    else:
        return int(lam*1.1)

@njit(fastmath=True, cache=True)
def cmp_lambda_small_mu(mu, nu):
    """
    Inverse of small-lambda CMP mean expansion
    up to O(mu^4).
    """

    a2 = 2.0**(-nu)
    a3 = 6.0**(-nu) 
    a4 = 24.0**(-nu)

    A = 2*a2 - 1
    B = 3*a3 - 3*a2 + 1
    C = 4*a4 - 4*a3 - 2*a2*a2 + 4*a2 - 1

    return (
        mu
        - A*mu**2
        + (2*A*A - B)*mu**3
        + (-5*A**3 + 5*A*B - C)*mu**4
    )

@njit(fastmath=True, cache=True)
def cmp_lambda_from_mean(mu, nu):
    """Rescale λ so CMP mean ≈ mu (Poisson mean)."""
    return (mu + (nu - 1) / (2 * nu)) ** nu


@njit(fastmath=True, cache=True)
def cmp_lambda_from_mu_allscale(mu, nu, sig=0.1):
    """Use small-lambda approximation for small mu, otherwise use large-lambda."""
    # The Poisson limit is exact, including below the blending threshold.
    if nu == 1.0:
        return mu
    # Threshold for switching between approximations
    threshold = 0.9  # Adjust this value as needed

    #  ---------- Adaptive blending ----------
    w = 0.5*(math.erf((mu - threshold)/sig)+1)
    mu_small = cmp_lambda_small_mu(mu, nu)
    mu_large = cmp_lambda_from_mean(mu, nu) if mu > 0.5 else 0 #np.nan_to_num does not work in numba!
    mu = (1-w)*mu_small + w*mu_large
    return mu


@njit(fastmath=True)
def cmp_sample(N_sat, nu): 
    """Draw by inverse CDF; even at nu=1 this uses a different RNG stream
    from np.random.poisson, so equal seeds do not imply equal catalogues.
    """
    lam_cmp = cmp_lambda_from_mu_allscale(N_sat, nu)
    max_x = get_max_x(N_sat)
    x = np.arange(max_x + 1) 
    logZ = cmp_logZ(lam_cmp, nu, max_x=max(300, int(max_x*1.1))) 
    pmf = cmp_pmf_vec(x, lam_cmp, nu, logZ) 
    cdf = np.cumsum(pmf) 
    cdf /= cdf[-1] 
    u = np.random.rand() 
    return (cdf < u).sum()


@njit(parallel=True, fastmath=True)
def compute_N(log10_Mh, fun_cHOD, fun_sHOD, p_cen, p_sat, nu=1, p_ab=None, Nthread=32, ab_arr=None, weight=None, conformity=False, link_sat_to_central=False, seed=None):
    """
    Compute the number of central and satellite galaxies in a halo using a HOD (Halo Occupation Distribution) model.

    Parameters:
        log10_Mh : ndarray
            Logarithmic mass of halos.
        fun_cHOD : callable
            Function to compute central occupation based on halo mass.
        fun_sHOD : callable
            Function to compute satellite occupation based on halo mass.
        p_cen : ndarray
            Parameters for the central HOD function.
        p_sat : ndarray
            Parameters for the satellite HOD function.
        p_ab : ndarray or None, optional
            Assembly bias parameters (central and satellite). Default is None.
        Nthread : int, optional
            Number of threads to use in parallel computation. Default is 32.
        ab_arr : ndarray or None, optional
            Assembly bias array per halo. Default is None.
        conformity : bool, optional
            Whether satellite number is correlated with central galaxy presence. Default is False.
        seed : ndarray or None, optional
            Seed array for RNG per thread. Default is None.

    Returns:
        Ncent : ndarray
            Expected number of central galaxies.
        N_sat : ndarray
            Expected number of satellite galaxies.
        cond_cent : ndarray
            Boolean array indicating presence of a central galaxy.
        proba_sat : ndarray
            Sampled number of satellites per halo.
    """

    # starting index of each thread
    # numba.set_num_threads(Nthread)
    hstart = np.rint(np.linspace(0, len(log10_Mh), Nthread + 1))
    cond_cent = np.zeros(log10_Mh.size, dtype=np.bool_)
    proba_sat = np.empty_like(log10_Mh, dtype=np.int64)
    Nb_sat = 0

    # figuring out the number of halos kept for each thread
    for tid in numba.prange(Nthread):
        if seed is not None:
            np.random.seed(seed[tid])
        for i in range(int(hstart[tid]), int(hstart[tid + 1])):
            Ncent = fun_cHOD(log10_Mh[i], p_cen)
            if p_ab is not None:
                Ncent *= (1 + np.sum(p_ab[0] * ab_arr[i])*(1-Ncent))
            cond_cent[i] = Ncent * weight[i] - np.random.uniform(0, 1) > 0
            if p_sat is not None :
                N_sat = fun_sHOD(log10_Mh[i], p_sat)
                if link_sat_to_central:
                    N_sat *= Ncent
                if p_ab is not None:
                    N_sat *= (1 + np.sum(p_ab[1] * ab_arr[i]))
                if N_sat <=0:
                    proba_sat[i] = 0
                elif conformity:
                    proba_sat[i] = np.random.poisson(N_sat*cond_cent[i])
                else :
                    if nu == 1:
                        proba_sat[i] = np.random.poisson(N_sat)
                    else:
                        proba_sat[i] = cmp_sample(N_sat, nu)
                if proba_sat[i] > 0:
                    Nb_sat += proba_sat[i]
                    
    return cond_cent, proba_sat, Nb_sat


@njit(fastmath=True, cache=True)
def _f_nfw(x):
    """
    NFW profile helper function.

    Parameters:
        x : float
            Input variable to compute NFW integral.

    Returns:
        float
            NFW integral value.
    """

    return np.log(1.+x)-x/(1.+x)



@njit(parallel=True, fastmath=True, cache=True)
def getPointsOnSphere_jit(nPoints, Nthread=32, seed=None):
    """
    Generate random points on a sphere surface using numba jit.

    Parameters:
        nPoints : int
            Number of points to generate.
        Nthread : int, optional
            Number of parallel threads to use. Default is 32.
        seed : ndarray or None
            Seed for RNG per thread. Default is None.

    Returns:
        ur : ndarray
            Array of shape (nPoints, 3) with unit vectors uniformly distributed on a sphere.
    """

    # numba.set_num_threads(Nthread)
    ind = min(Nthread, nPoints)
    # starting index of each thread
    hstart = np.rint(np.linspace(0, nPoints, ind+1))
    ur = np.zeros((nPoints, 3), dtype=np.float64)    
    cmin = -1
    cmax = +1

    for tid in numba.prange(Nthread):
        if seed is not None:
            np.random.seed(seed[tid])
        for i in range(hstart[tid], hstart[tid + 1]):
            u1, u2 = np.random.uniform(0, 1), np.random.uniform(0, 1)
            ra = 0 + u1*(2*np.pi-0)
            dec = np.pi - (np.arccos(cmin+u2*(cmax-cmin)))

            ur[i, 0] = np.sin(dec) * np.cos(ra)
            ur[i, 1] = np.sin(dec) * np.sin(ra)
            ur[i, 2] = np.cos(dec)
    return ur



@njit(parallel=True, fastmath=True, cache=True)
def compute_ngal(log10_Mh, fun_cHOD, fun_sHOD, p_cen, p_sat=None, conformity=False, link_sat_to_central=False):
    """
    Compute total number of galaxies and satellite fraction from a given HOD parameter set.

    Parameters:
        log10_Mh : ndarray
            Logarithmic halo mass.
        fun_cHOD : callable
            Function to compute central occupation.
        fun_sHOD : callable
            Function to compute satellite occupation.
        Nthread : int
            Number of threads for parallel computation.
        p_cen : ndarray
            Parameters for central occupation.
        p_sat : ndarray or None, optionnal
            Parameters for satellite occupation. Default is None.
        conformity : bool, optionnal
            Use conformity when calculating satellites. Default is False.

    Returns:
        tuple(float, float, float)
            Total number of galaxies, satellite fraction, and mean halo mass of the sample.
    """

    # numba.set_num_threads(Nthread)
    nbinsM, logbinM = np.histogram(log10_Mh, bins=100)[:2]
    ngal_c = np.empty_like(nbinsM, dtype=np.float64)
    ngal_sat = np.empty_like(nbinsM, dtype=np.float64)
    mass_b = np.empty_like(nbinsM, dtype=np.float64)
    if p_sat is not None :
        for i in numba.prange(len(logbinM)):
            LogM = ((logbinM[i]+logbinM[i+1])*0.5)
            Ncent = fun_cHOD(LogM, p_cen)
            if Ncent < 0:
                Ncent = 0
            if LogM < p_sat[1]:
                N_sat = 0
            else:
                N_sat = fun_sHOD(LogM, p_sat)
            ngal_c[i] = (nbinsM[i] * Ncent)
            ngal_sat[i] = (nbinsM[i]* Ncent*N_sat) if conformity | link_sat_to_central else (nbinsM[i] * N_sat)
            mass_b[i] = (ngal_c[i]+ngal_sat[i])*10**LogM
        ngal_tot = ngal_c.sum()+ngal_sat.sum()
        return ngal_tot, ngal_sat.sum()/ngal_tot, mass_b.sum()/ngal_tot
    else: 
        for i in numba.prange(len(logbinM)):
            LogM = ((logbinM[i]+logbinM[i+1])*0.5)
            Ncent = fun_cHOD(LogM, p_cen)
            if Ncent < 0:
                Ncent = 0
            ngal_c[i] = (nbinsM[i] * Ncent)
            mass_b[i] = (ngal_c[i]*10**LogM)
        return ngal_c.sum(), 0, mass_b.sum()/(ngal_c.sum())

"""Interpolation of lambertW function for NFW computation"""
xt = np.linspace(-1,0,1000000)
ft = np.real(scipy.special.lambertw(xt, k=0))
interp_lambertw= njit(fastmath=True)(lambda x: np.interp(x, xt, ft))

@njit(fastmath=True)
def get_etavir_nfw(c): 
    """
    Draw a normalized NFW radius using inversion sampling of the cumulative mass profile.
    This use analytic description of the NFW profile from https://arxiv.org/pdf/1805.09550
    Parameters:
        c : float
            Concentration parameter of the NFW profile.

    Returns:
        float
            Normalized radius (r/Rvir).
    """

    rd_u = np.random.uniform() * (np.log(1.0 + c)-c/(1.0 + c))  
    return (-(1.0/interp_lambertw(-np.exp(-rd_u-1)))-1)/c

@njit(parallel=True, fastmath=True)
def compute_fast_NFW(x_h, y_h, z_h, vx_h, vy_h, vz_h, c, M, Rvir, rd_pos, rd_vel, exp_frac=0, exp_scale=1, nfw_rescale=1, vrms_h=None, f_sigv=None, v_infall=None, vel_sat='NFW', Nthread=32, seed=None):
    
    """
    Compute satellite galaxy positions and velocities using the NFW profile.

    Parameters:
        x_h, y_h, z_h : ndarray
            Host halo positions.
        vx_h, vy_h, vz_h : ndarray
            Host halo velocities.
        c : ndarray
            Host halo concentration parameters.
        M : ndarray
            Host halo masses.
        Rvir : ndarray
            Host halo radii.
        rd_pos, rd_vel : ndarray
            Random vectors for position and velocity.
        exp_frac : float, optional
            Fraction of satellites to sample using an exponential halo profile. Default is 0.
        exp_scale : float, optional
            Scale of exponential halo profile. Default is 1.
        nfw_rescale : float, optional
            Rescaling factor of the concentration parameter.  Default is 1.
        vrms_h : ndarray, optional
            RMS velocity of hots halo particles.  Need to be definied if ``vel_sat`` is 'rd_normal' or 'infall'. Default is None.
        f_sigv : float, optional
            Velocity dispersion factor. Default is 1.
        v_infall : float, optional
            Additional infall velocity toward the host halo center in km/s. Need to be definied if ``vel_sat`` is 'infall'. Default is None.
        vel_sat : str, optional
            Velocity model: 'NFW', 'rd_normal', or 'infall'.
        Nthread : int, optional
            Number of threads for computation. Default is 32.
        seed : ndarray or None, optional
            RNG seeds per thread. Default is None.

    Returns:
        tuple of ndarrays
            Satellite positions and velocities (x, y, z, vx, vy, vz).
    """

    
    # numba.set_num_threads(Nthread)
    G = 4.302e-6  # in kpc/Msol (km.s)^2
    # Validate vel_sat once, before the prange loop. Keeping the raise inside the
    # loop gives it a second exit path, which prevents Numba from parallelizing it.
    
    x_sat = np.empty_like(x_h)
    y_sat = np.empty_like(y_h)
    z_sat = np.empty_like(z_h)
    vx_sat = np.empty_like(vx_h)
    vy_sat = np.empty_like(vy_h)
    vz_sat = np.empty_like(vz_h)

    # starting index of each thread
    hstart = np.rint(np.linspace(0, x_h.size, Nthread + 1))
    for tid in numba.prange(Nthread):
        if seed is not None:
            np.random.seed(seed[tid])
        for i in range(int(hstart[tid]), int(hstart[tid + 1])):
            if np.random.uniform(0,1) < exp_frac:
                tt = np.random.exponential(scale=exp_scale)
                etaVir = tt/c[i]
            else:
                etaVir = get_etavir_nfw(c[i])*nfw_rescale    
            
            p = etaVir * Rvir[i] / 1000
            x_sat[i] = x_h[i] + rd_pos[i, 0] * p
            y_sat[i] = y_h[i] + rd_pos[i, 1] * p
            z_sat[i] = z_h[i] + rd_pos[i, 2] * p
            if vel_sat == 'NFW':
                v = np.sqrt(G*M[i]/Rvir[i]) * \
                            np.sqrt(_f_nfw(c[i] * etaVir) / (etaVir * _f_nfw(c[i])))
                vx_sat[i] = vx_h[i] + rd_vel[i, 0] * v *f_sigv
                vy_sat[i] = vy_h[i] + rd_vel[i, 1] * v *f_sigv
                vz_sat[i] = vz_h[i] + rd_vel[i, 2] * v *f_sigv
            elif (vel_sat == 'rd_normal'):
                sig = vrms_h[i]*0.577*f_sigv
                vx_sat[i] = np.random.normal(loc=vx_h[i], scale=sig)
                vy_sat[i] = np.random.normal(loc=vy_h[i], scale=sig)
                vz_sat[i] = np.random.normal(loc=vz_h[i], scale=sig)
            else:  # vel_sat == 'infall' (validated before the loop)
                sig = vrms_h[i]*0.577
                norm = np.sqrt((x_h[i] - x_sat[i])**2 + (y_h[i] - y_sat[i])**2 + (z_h[i] -z_sat[i])**2)
                v_r = np.random.normal(loc=v_infall, scale=sig)
                vx_sat[i] = np.random.normal(loc=vx_h[i], scale=sig) + ((x_h[i] - x_sat[i])/norm * v_r)
                vy_sat[i] = np.random.normal(loc=vy_h[i], scale=sig) + ((y_h[i] - y_sat[i])/norm * v_r)
                vz_sat[i] = np.random.normal(loc=vz_h[i], scale=sig) + ((z_h[i] - z_sat[i])/norm * v_r)
    return x_sat, y_sat, z_sat, vx_sat, vy_sat, vz_sat



def plot_HOD(p_cen, p_sat, fun_cHOD, fun_sHOD, logM = np.linspace(10.8,15,100), fig=None, color=None, label=None, figsize=(5,4), show=True):
    
    """
    Plot HOD curves for central and satellite galaxies.

    Parameters:
        p_cen : ndarray
            Central HOD parameters.
        p_sat : ndarray
            Satellite HOD parameters.
        fun_cHOD : callable
            Central HOD function.
        fun_sHOD : callable
            Satellite HOD function.
        logM : ndarray, optional
            Mass bins for evaluation. Default is numpy.linspace(10.8,15,100).
        fig : matplotlib.figure.Figure or None, optional
            Existing figure to plot into. Default is None.
        color : str or None, optional
            Line color. Default is None.
        label : str or None, optional
            Legend label. Default is None.
        figsize : tuple, optional
            Figure size. Default is (5,4).
        show : bool
            Whether to show the plot immediately. Default is True.

    Returns:
        matplotlib.figure.Figure
            The plotted figure.
    """
    import matplotlib.pyplot as plt


    if fig is None:
        fig, ax = plt.subplots(1,1, figsize=figsize)
        ax.set_yscale('log')
        ax.axhline(y=1, ls='--', c='grey')
        ax.set_ylim(0.001,10)
        ax.set_ylabel('$<N_{gal}>$')
        ax.set_xlabel('$\log(M_h\ [M_{\odot}])$')
    else:
        ax= fig.axes[0] 
    
    i = int((len(ax.get_lines())-1)/2)
    color= f'C{i}' if color is None else color
    cen = [fun_cHOD(lM, p_cen) for lM in logM]
    sat = np.nan_to_num([fun_sHOD(lM, p_sat) for lM in logM])
    if i ==0:
        ax.plot(logM, cen, lw=1.2, color='k', label='Central')
        ax.plot(logM, sat, lw=1.2, ls='--', color='k', label='Satellite')
    ax.plot(logM, cen, lw=1.2, color=color, label=label)
    ax.plot(logM, sat, lw=1.2, ls='--', color=color)
    ax.legend(loc='upper right')
    if show:
        fig.tight_layout()
        plt.show()
    return fig




import time
from numba import objmode
@njit(parallel=True, fastmath=True, cache=True)
def halo_to_particle_indices(sorted_hid, order_idx, halo_id_list, Nthread=32):
    """
    Build a CSR-style index mapping each halo_id to all corresponding particle indices. (AI based code)

    Parameters
    ----------
    particle_halo_id : ndarray[int64]
        Array of shape (N_particles,). For each particle, gives the host halo ID.
        May contain repeated halo IDs.

    halo_id_list : ndarray[int64]
        Array of unique halo IDs for which to retrieve particle indices.
        Order is preserved in the output (CSR corresponds to this order).

    Returns
    -------
    flat : ndarray[int64]
        Flattened array of all particle indices corresponding to each halo ID
        (concatenated CSR style).

    offsets : ndarray[int64]
        Array of shape (len(halo_id_list) + 1,). For halo `i`,
        particle indices are in `flat[offsets[i]:offsets[i+1]]`.
    Nthread : int, optional
        Number of threads for parallel computation. Default is 32.

    Notes
    -----
    - Works efficiently for >100M particles.
    - Memory-efficient (no dense array indexed by max ID).
    - Fully deterministic (stable sorting).
    - Parallelized with `numba.prange`.

    Example
    -------
    >>> Np = 10_000_000
    >>> halo_id_list = np.arange(100_000, dtype=np.int64)
    >>> particle_halo_id = np.random.choice(halo_id_list, Np)
    >>> flat, offsets = halo_to_particle_indices(particle_halo_id, halo_id_list, Nthread=32)
    >>> # indices for first halo
    >>> flat[offsets[0]:offsets[1]]
    """
    # numba.set_num_threads(Nthread)
    Np = len(sorted_hid)
    Nh = len(halo_id_list)

    # === Phase 2: identify group boundaries for unique halo IDs
    with objmode(st='f8'):
        st = time.perf_counter()
    uniq_hid = []
    start_pos = []
    last_val = sorted_hid[0]
    uniq_hid.append(last_val)
    start_pos.append(0)
    for i in range(1, Np):
        val = sorted_hid[i]
        if val != last_val:
            uniq_hid.append(val)
            start_pos.append(i)
            last_val = val
    start_pos.append(Np)

    uniq_hid = np.array(uniq_hid, dtype=np.int64)
    start_pos = np.array(start_pos, dtype=np.int64)
    Nuniq = len(uniq_hid)
    with objmode():
        print('Grouping time (s):', time.perf_counter()-st, flush=True)

    # === Phase 3: binary search each target halo_id in uniq_hid
    with objmode(st='f8'):
        st = time.perf_counter()
    start_match = np.full(Nh, -1, dtype=np.int64)
    end_match = np.full(Nh, -1, dtype=np.int64)

    for i in numba.prange(Nh):
        v = halo_id_list[i]
        lo = 0
        hi = Nuniq - 1
        found = -1
        while lo <= hi:
            mid = (lo + hi) // 2
            if uniq_hid[mid] == v:
                found = mid
                break
            elif uniq_hid[mid] < v:
                lo = mid + 1
            else:
                hi = mid - 1
        if found != -1:
            start_match[i] = start_pos[found]
            end_match[i] = start_pos[found + 1]
    with objmode():
        print('Matching time (s):', time.perf_counter()-st, flush=True)
    # === Phase 4: compute offsets (CSR)
    with objmode(st='f8'):
        st = time.perf_counter()
    offsets = np.empty(Nh + 1, dtype=np.int64)
    offsets[0] = 0
    for i in range(Nh):
        if start_match[i] != -1:
            count = end_match[i] - start_match[i]
        else:
            count = 0
        offsets[i + 1] = offsets[i] + count
    with objmode():
        print('Offsets time (s):', time.perf_counter()-st, flush=True)
    # === Phase 5: fill flat array

    with objmode(st='f8'):
        st = time.perf_counter()
    flat = np.empty(offsets[-1], dtype=np.int64)
    for i in numba.prange(Nh):
        if start_match[i] != -1:
            start = start_match[i]
            end = end_match[i]
            out_start = offsets[i]
            out_end = offsets[i + 1]
            flat[out_start:out_end] = order_idx[start:end]
    with objmode():
        print('Filling time (s):', time.perf_counter()-st, flush=True)
    return flat, offsets


@njit(fastmath=True, parallel=True)
def sample_satellites_from_particles(xp, yp, zp, vxp, vyp, vzp,
                                     flat, offsets,
                                     nb_sat, seed=None, vx_h=None, vy_h=None,
                                     vz_h=None, f_sigv=1.0):
    """
    Randomly sample satellite positions and velocities from particle data
    using precomputed CSR-style halo-to-particle mapping.

    Parameters
    ----------
    xp, yp, zp : ndarray
        Particle positions.
    vxp, vyp, vzp : ndarray
        Particle velocities.
    flat, offsets : ndarray[int64]
        CSR structure giving the indices of particles per halo.
        For halo i: flat[offsets[i]:offsets[i+1]] gives its particle indices.
    nb_sat : ndarray[int64]
        Number of satellites to draw per halo.
    seed : ndarray[int64], optional
        One random seed per host halo for reproducibility.
    vx_h, vy_h, vz_h : ndarray, optional
        Host halo velocities in CSR halo order; required for f_sigv other than one.
    f_sigv : float, optional
        Apply v_sat = v_halo + f_sigv * (v_part - v_halo) in each component.
        One preserves the particle velocities.

    Returns
    -------
    x_sat, y_sat, z_sat, vx_sat, vy_sat, vz_sat : ndarray[float32]
        Positions and velocities of all selected satellites (flattened).
        Length = sum(nb_sat)
    mask_missing : ndarray[bool]
        Boolean mask of length sum(nb_sat), True if the corresponding
        satellite needs to be generated artificially (not enough particles).

    Notes
    -----
    - Parallelized across halos.
    - If a halo has fewer particles than nb_sat[i], existing particles are
      all used and the remainder are marked as missing.
    - Designed for 100M+ particles and 10M+ halos.
    """

    # numba.set_num_threads(Nthread)
    Nh = len(nb_sat)
    Nsat_total = np.sum(nb_sat)
    if f_sigv != 1.0 and (vx_h is None or vy_h is None or vz_h is None):
        raise ValueError('Host halo velocities are required for particle velocity bias.')

    mask_nfw = np.zeros(Nsat_total, dtype=np.bool_)
    x_sat = np.empty(Nsat_total, dtype=np.float32)
    y_sat = np.empty(Nsat_total, dtype=np.float32)
    z_sat = np.empty(Nsat_total, dtype=np.float32)
    vx_sat = np.empty(Nsat_total, dtype=np.float32)
    vy_sat = np.empty(Nsat_total, dtype=np.float32)
    vz_sat = np.empty(Nsat_total, dtype=np.float32)

    # Build cumulative sum to know where to write per halo
    cum_sum = np.zeros(Nh + 1, dtype=np.int64)
    for i in range(Nh):
        cum_sum[i + 1] = cum_sum[i] + nb_sat[i]

    for i in numba.prange(Nh):
        start_p = offsets[i]
        end_p = offsets[i + 1]
        n_p = end_p - start_p  # number of particles for this halo
        n_s = nb_sat[i]        # number of satellites to sample
        out_start = cum_sum[i]
        out_end = cum_sum[i + 1]

        if n_p == 0:
            # No particles in this halo — all satellites missing
            mask_nfw[out_start:out_end] = True
            continue
        # Seed RNG per halo for reproducibility
        if seed is not None:
            np.random.seed(seed[i])
        if n_p >= n_s:
            # Enough particles — draw without replacement
            idx_local = np.random.choice(n_p, n_s, replace=False)
            indices = flat[start_p:end_p][idx_local]
        else:
            # Not enough particles — take all, fill rest as missing
            indices = flat[start_p:end_p]
            mask_nfw[out_start + n_p:out_end] = True

        # Fill in selected satellites
        n_fill = min(n_p, n_s)
        for j in range(n_fill):
            particle_index = indices[j]
            satellite_index = out_start + j
            x_sat[satellite_index] = xp[particle_index]
            y_sat[satellite_index] = yp[particle_index]
            z_sat[satellite_index] = zp[particle_index]
            if vx_h is not None and vy_h is not None and vz_h is not None:
                vx_sat[satellite_index] = vx_h[i] + f_sigv * (vxp[particle_index] - vx_h[i])
                vy_sat[satellite_index] = vy_h[i] + f_sigv * (vyp[particle_index] - vy_h[i])
                vz_sat[satellite_index] = vz_h[i] + f_sigv * (vzp[particle_index] - vz_h[i])
            else:
                vx_sat[satellite_index] = vxp[particle_index]
                vy_sat[satellite_index] = vyp[particle_index]
                vz_sat[satellite_index] = vzp[particle_index]

    return x_sat, y_sat, z_sat, vx_sat, vy_sat, vz_sat, mask_nfw



def update_dic(d, u):
    """
    Update a nested dictionary `d` with another dictionary `u`, merging values recursively.
    """
    import collections.abc
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update_dic(d.get(k, {}), v)
        else:
            d[k] = v
    return d

@njit(parallel=True)
def _assign_bin_values(bin_idx, col, bins, out):
    """
    Assign scaled values to elements within each bin, in parallel.

    Parameters
    ----------
    bin_idx : ndarray of shape (N,)
        Integer array giving the bin index of each element.
    col : ndarray of shape (N,)
        Array of values used for sorting within each bin.
    bins : ndarray of shape (nbins+1,)
        The bin edges used to compute `bin_idx`.
    out : ndarray of shape (N,)
        Preallocated output array to be filled in place.

    Notes
    -----
    - For each bin, elements are sorted by `col` in descending order.
    - Within a bin of size `m`, values are assigned as
      linspace(-0.5, 0.5, m).
    - Runs in parallel across bins using `numba.prange`.
    """
    nbins = len(bins) - 1
    n = len(col)

    for b in numba.prange(nbins):
        # collect indices of elements in this bin
        idx_in_bin = []
        for i in range(n):
            if bin_idx[i] == b:
                idx_in_bin.append(i)
        if len(idx_in_bin) == 0:
            continue

        idx_in_bin = np.array(idx_in_bin)

        # argsort by col descending
        vals = col[idx_in_bin]
        order = np.argsort(vals)[::-1]

        # assign linspace
        size = len(order)
        step = 1.0 / (size - 1) if size > 1 else 0.0
        for j in range(size):
            out[idx_in_bin[order[j]]] = -0.5 + j * step


def initialize_assembly_bias_value(logMh, col, bins=50):
    """
    Vectorized and parallelized version of binning + ranking.

    Parameters
    ----------
    logMh : ndarray of shape (N,)
        Array of values to be binned (e.g., log10 halo masses).
    col : ndarray of shape (N,)
        Values used for ranking within bins.
    bins : int or array-like, optional (default=50)
        If int, the number of bins. If array, explicit bin edges.

    Returns
    -------
    out : ndarray of shape (N,)
        Output array where each element has been assigned a value
        in [-0.5, 0.5] based on its rank within its bin.

    Notes
    -----
    - Uses `np.histogram_bin_edges` if `bins` is an integer.
    - Bin assignment is done with `np.digitize`.
    - The heavy computation is offloaded to `_assign_bin_values`
      (Numba parallel kernel).
    - Designed to scale to very large arrays (10^7–10^8 elements).
    """
    # ensure bins are edges
    if np.isscalar(bins):
        bins = np.histogram_bin_edges(logMh, bins=bins)

    # assign bin indices
    bin_idx = np.digitize(logMh, bins) - 1
    out = np.zeros_like(logMh)

    # fill in parallel
    _assign_bin_values(bin_idx, col, bins, out)

    return out
