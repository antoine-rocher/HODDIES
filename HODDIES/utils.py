import numpy as np
try:
    from pycorr import TwoPointCorrelationFunction, project_to_multipoles, project_to_wp

except ImportError:
    import warnings
    warnings.warn(
        'Could not import pycorr. Install pycorr with ' \
        '"python -m pip install git+https://github.com/cosmodesi/pycorr#egg=pycorr[corrfunc]".' \
        'pycorr currently use a branch of Corrfunc, uninstall previous Corrfunc version (if any): "pip uninstall Corrfunc"'\
        '' 
    )

from numba import njit, numba
import os
# import mpytools as mpy
import scipy


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
    vsmear : float, optional
        Add redshift error using gaussian distribution in km/s. Default is 0.

    Returns
    -------
    pos_rsd : list of ndarray
        List of arrays containing RSD-applied positions for x, y, and z.
    """

    rsd_factor = 1 / (1 / (1 + z) * H_0 * cosmo.efunc(z))
    pos_rsd = [cat[p] % boxsize if p !=los else (cat[p] + (cat['v'+p] + np.random.normal(0,vsmear, size=len(cat[p])))*rsd_factor) %boxsize if vsmear is not None else (cat[p] + cat['v'+p]*rsd_factor) %boxsize for p in 'xyz']
    return pos_rsd
    
    
def compute_2PCF(pos1, edges, boxsize, ells=(0, 2), los='z', nthreads=32, R1R2=None, pos2=None, mpicomm=None):

    """
    Compute the 2-point correlation function multipoles for a periodic box.

    Parameters
    ----------
    pos1 : array-like
        Positions of sample 1 (e.g., galaxies or halos).
    edges : list of arrays
        Bin edges in separation (s, mu).
    boxsize : float
        Size of the simulation box.
    ells : tuple of int, optional
        Multipoles to project onto. Default is (0, 2).
    los : {'x', 'y', 'z'}, optional
        Line-of-sight direction. Default is 'z'.
    nthreads : int, optional
        Number of threads for parallel computation. Default is 32.
    R1R2 : array-like, optional
        Precomputed RR counts for normalization. Default is None.
    pos2 : array-like, optional
        Positions of sample 2 (for cross-correlations). Default is None.
    mpicomm : object, optional
        MPI communicator. Default is None.

    Returns
    -------
    s, (multipoles) : tuple(array, ndarray)
        Average separation for each s bin and computed multipoles of the 2PCF.
    """    

    result = TwoPointCorrelationFunction('smu', edges, 
                                         data_positions1=pos1, data_positions2=pos2, engine='corrfunc', 
                                         boxsize=boxsize, los=los, nthreads=nthreads, R1R2=R1R2, mpicomm=mpicomm)
    
    return project_to_multipoles(result, ells=ells)

    
def compute_wp(pos1, edges, boxsize, pimax=40, los='z', nthreads=32, R1R2=None, pos2=None, mpicomm=None):
    """
    Compute the projected correlation function w_p(r_p).

    Parameters
    ----------
    pos1 : array-like
        Positions of sample 1 (e.g., galaxies or halos).
    edges : list of arrays
        Bin edges for projected separation (r_p, pi).
    boxsize : float
        Size of the simulation box.
    pimax : float, optional
        Maximum line-of-sight separation for integration.
    los : {'x', 'y', 'z'}, optional
        Line-of-sight direction. Default is 'z'.
    nthreads : int, optional
        Number of threads for parallel computation. Default is 32.
    R1R2 : array-like, optional
        Precomputed RR counts for normalization. Default is None.
    pos2 : array-like, optional
        Positions of sample 2 for cross-correlations. Default is None.
    mpicomm : object, optional
        MPI communicator. Default is None.

    Returns
    -------
    rp, wp : tuple(array, array)
        Seperation and projected correlation function.
    """

        
    result = TwoPointCorrelationFunction('rppi', edges, 
                                         data_positions1=pos1, data_positions2=pos2, engine='corrfunc', 
                                         boxsize=boxsize, los=los, nthreads=nthreads, R1R2=R1R2, mpicomm=mpicomm)
    
    return project_to_wp(result, pimax=pimax)


@njit(parallel=True, fastmath=True)
def compute_N(log10_Mh, fun_cHOD, fun_sHOD, p_cen, p_sat, p_ab=None, Nthread=32, ab_arr=None, conformity=False, seed=None):
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

    
    numba.set_num_threads(Nthread)
    # starting index of each thread
    hstart = np.rint(np.linspace(0, len(log10_Mh), Nthread + 1))
    Ncent = np.empty_like(log10_Mh)
    cond_cent = np.empty_like(log10_Mh)
    N_sat = np.empty_like(log10_Mh)
    proba_sat = np.empty_like(log10_Mh, dtype=np.int64)
    
    # figuring out the number of halos kept for each thread
    for tid in numba.prange(Nthread):
        if seed is not None:
            np.random.seed(seed[tid])
        for i in range(int(hstart[tid]), int(hstart[tid + 1])):
            M_0 = p_sat[1]
            Ncent[i] = fun_cHOD(log10_Mh[i], p_cen)
            if p_ab is not None:
                Ncent[i] *= (1 + np.sum(p_ab[0] * ab_arr[i])*(1-Ncent[i]))
            cond_cent[i] = Ncent[i] - np.random.uniform(0, 1) > 0
            if p_sat is not None :
                M_0 = p_sat[1]
                if log10_Mh[i] <= M_0:
                    N_sat[i] = 0
                elif np.abs(log10_Mh[i]- M_0) < -0.001:
                    p_sat[1] += 0.001
                    N_sat[i] = fun_sHOD(log10_Mh[i], p_sat)
                    p_sat[1] -= 0.001
                else:
                    N_sat[i] = fun_sHOD(log10_Mh[i], p_sat)

                if p_ab is not None:
                    N_sat[i] *= (1 + np.sum(p_ab[1] * ab_arr[i])*(1-N_sat[i]))
                if conformity:
                    proba_sat[i] = np.random.poisson(N_sat[i]*cond_cent[i])
                else :
                    proba_sat[i] = np.random.poisson(N_sat[i])
                
    return Ncent, N_sat, cond_cent, proba_sat


@njit(fastmath=True)
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


@njit(fastmath=True)
def _rescale(a, b):
    """
    Rescale vectors by another vector using broadcasting.

    Parameters:
        a : ndarray
            Input array to be scaled.
        b : ndarray
            Scaling array.

    Returns:
        ndarray
            Scaled array.
    """

    return np.transpose(np.multiply(np.transpose(a), np.transpose(b)))

@njit(parallel=True, fastmath=True)
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

    numba.set_num_threads(Nthread)
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



def rd_draw_NFW(nPoints, burn_in=100000):
    """
    Draw random samples from a 3D NFW profile using the Metropolis-Hastings algorithm.

    Parameters:
        nPoints : int
            Total number of samples to generate.
        nPoints : int, optional
            Number of point to remove from the Metropolis-Hastings algorithm. Default is 100000.

    Returns:
        ndarray
            Random samples from the NFW profile.
    """

    
    if nPoints <= burn_in:
        raise ValueError (f'Error : NPoints must be above {burn_in}')
    epsilon = 0.3
    previousX = 0.3

    def NFWprofile(x):
        # multiply by x^2 to get P(r) and not rho(r)
        return 1./(x*(1+x)**2)
    previousP = NFWprofile(previousX)
    data = np.zeros(nPoints)
    i = 0
    for step in np.arange(nPoints)+1:
        evalX = previousX+2.*(np.random.uniform()-0.5)*epsilon
        evalP = NFWprofile(evalX)
        if evalX < 0.:
            evalP = 0.
        elif evalX < 0.01:
            evalP = NFWprofile(0.01)
        else:
            pass
        R = evalP/previousP
        if R >= 1:
            previousX = evalX*1.
            previousP = evalP*1.
        else:
            if np.random.uniform() < R:
                previousX = evalX*1.
                previousP = evalP*1.
        data[i] = previousX*1.
        i += 1
        # if step%100000==0 and step>1000 :
        #     print (step)
        #     print (np.max(data))
        #     x=np.linspace(0.,40.,1000)
        #     h,e=np.histogram(data[data>0],bins=1000,range=(0.,40.))
        #     plt.plot(x,h)
        #     x=np.linspace(0.,40.,1000)
        #     plt.plot(x,21000.*4.*NFWprofile(x))       #Normalisation a revoir mais profile OK
        #     plt.xscale('log')
        #     plt.yscale('log')
        #     plt.show()
    dataPruned = data[burn_in:]
    np.random.shuffle(dataPruned)
    os.makedirs('data', exist_ok=True)
    np.save('data/nfw.npy', dataPruned)
    return dataPruned



@njit(parallel=True, fastmath=True)
def compute_ngal(log10_Mh, fun_cHOD, fun_sHOD, Nthread, p_cen, p_sat=None, conformity=False):
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
        tuple(float, float)
            Total number of galaxies, and satellite fraction.
    """

    nbinsM, logbinM = np.histogram(log10_Mh, bins=100)[:2]
    LogM = np.zeros(len(logbinM)-1)
    dM = np.diff(logbinM)[0]
    ngal_c = 0
    ngal_sat = 0 
    # starting index of each thread
    hstart = np.rint(np.linspace(0, len(logbinM)-1, Nthread + 1))
    #p_sat=None
    if p_sat is not None :
        for tid in numba.prange(Nthread):
            for i in range(int(hstart[tid]), int(hstart[tid + 1])):
                LogM = ((logbinM[i]+logbinM[i+1])*0.5)
                Ncent = fun_cHOD(LogM, p_cen)
                if Ncent < 0:
                    Ncent = 0
                if LogM < p_sat[1]:
                    N_sat = 0
                else:
                    N_sat = fun_sHOD(LogM, p_sat)
                ngal_c += (nbinsM[i]/dM * Ncent*dM)
                ngal_sat += (nbinsM[i]/dM * Ncent*N_sat*dM) if conformity else (nbinsM[i]/dM * N_sat*dM)
        ngal_tot = ngal_c+ngal_sat
        return ngal_tot, ngal_sat/ngal_tot
    else : 
        for tid in numba.prange(Nthread):
            for i in range(int(hstart[tid]), int(hstart[tid + 1])):
                LogM = ((logbinM[i]+logbinM[i+1])*0.5)
                Ncent = fun_cHOD(LogM, p_cen)
                if Ncent < 0:
                    Ncent = 0
                ngal_c += (nbinsM[i]/dM * Ncent*dM)
        
        return ngal_c, 0
    
"""Interpolation of lambertw function for NFW computation"""
xt = np.linspace(-1,0,1000000)
ft = np.real(scipy.special.lambertw(xt, k=0))
interp_lambertw= njit(parallel=True, fastmath=True)(lambda x: np.interp(x, xt, ft))

@njit(fastmath=True)
def get_etavir_nfw(c): 
    """
    Draw a normalized NFW radius using inversion sampling of the cumulative mass profile.

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

    
    numba.set_num_threads(Nthread)
    G = 4.302e-6  # in kpc/Msol (km.s)^2
    # figuring out the number of halos kept for each thread
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
            ind = i
            #while (NFW_draw[ind] > c[i]):
            #    ind = np.random.randint(0, len(NFW_draw))
            #etaVir = NFW_draw[ind]/c[i]  # =r/rvir
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
            elif (vel_sat == 'rd_normal') | (vel_sat=='infall'):
                sig = vrms_h[i]*0.577*f_sigv
                vx_sat[i] = np.random.normal(loc=vx_h[i], scale=sig)
                vy_sat[i] = np.random.normal(loc=vy_h[i], scale=sig)
                vz_sat[i] = np.random.normal(loc=vz_h[i], scale=sig)
                if vel_sat=='infall':
                    norm = np.sqrt((x_h[i] - x_sat[i])**2 + (y_h[i] - y_sat[i])**2 + (z_h[i] -z_sat[i])**2)
                    v_r = np.random.normal(loc=v_infall, scale=sig)
                    vx_sat[i] += (x_h[i] - x_sat[i])/norm * v_r
                    vy_sat[i] += (y_h[i] - y_sat[i])/norm * v_r
                    vz_sat[i] += (z_h[i] - z_sat[i])/norm * v_r
            else:
                raise ValueError(
                    'Wrong vel_sat argument only "rd_normal", "infall", "NFW"')
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


def compute_power_spectrum(pos1, boxsize, kedges, pos2=None, los='z', nmesh=256, resampler='tsc', interlacing=2, ells=(0, 2), mpicomm=None):
    """
    Compute the power spectrum multipoles from a catalog using FFT-based methods.

    Parameters
    ----------
    pos1 : array-like
        Positions of catalog 1.
    boxsize : float
        Size of the simulation box.
    kedges : tuple
        k-bin edges for the power spectrum.
    pos2 : array-like, optional
        Positions of catalog 2 (for cross-spectrum).
    los : array-like, optional
        Line-of-sight direction.
    nmesh : int, optional
        Number of mesh cells per dimension. Default is 256.
    resampler : str, optional
        Mass assignment scheme. Default is 'tsc'.
    interlacing : int, optional
        Interlacing order for FFT. Default is 2.
    ells : tuple of int, optional
        Multipoles to compute. Default is (0, 2, 4).
    mpicomm : object, optional
        MPI communicator.

    Returns
    -------
    array
        Power spectrum multipoles.
    """
    from pypower import CatalogFFTPower

    result = CatalogFFTPower(
        data_positions1=pos1, data_positions2=pos2,
        boxsize=boxsize, nmesh=nmesh, kedges=kedges,
        los=los, resampler=resampler,
        interlacing=interlacing,
        position_type='pos', ells=ells,
        mpicomm=mpicomm
    )
    return result.poles

# #### MPI FUNCTIONS
# Need to be tested

# def compute_ngal_mpi(log10_Mh, fun_cHOD, fun_sHOD, p_cen, p_sat=None):
#     """
#     --- Compute the number of galaxy and the satelitte fraction form HOD parameters
#     """
    
#     nbinsM, logbinM = np.histogram(log10_Mh, bins=100, range=(10, 16))[:2]
#     dM = np.diff(logbinM)[0]
#     LogM = 0.5*(logbinM[:-1]+logbinM[1:])

#     Ncent = fun_cHOD(LogM, p_cen)
#     Ncent[Ncent < 0] = 0
#     ngal_c = (nbinsM/dM * Ncent*dM).sum()

#     if p_sat is not None:
#         M_0 = p_sat[1]
#         Nsat = np.zeros(100)
#         Nsat[LogM > M_0] = fun_sHOD(LogM[LogM > M_0], p_sat)
#         ngal_sat = (nbinsM/dM * Nsat*dM).sum()
#     else:
#         ngal_sat = 0
#     return ngal_c, ngal_sat


# def compute_N_mpi(log10_Mh, fun_cHOD, fun_sHOD, p_cen, p_sat=None, p_ab=None, ab_arr=None):

#     """
#     --- Compute the probability N for central galaxies given a HOD model
#     """
    
#     Ncent = fun_cHOD(log10_Mh, p_cen)
#     if p_ab is not None:
#         Ncent *= (1 + np.sum(p_ab[0] * ab_arr)*(1-Ncent))
#     if p_sat is None:
#         return Ncent, 0
#     M_0 = p_sat[1]
#     N_sat = np.zeros_like(log10_Mh)
#     N_sat[log10_Mh > M_0] = fun_sHOD(log10_Mh[log10_Mh > M_0], p_sat)
#     if p_ab is not None:
#         N_sat *= (1 + np.sum(p_ab[1] * ab_arr)*(1-N_sat))
#     return Ncent, N_sat




# def getPointsOnSphere_mpi(nPoints, rng):
#         u1, u2 = rng.uniform(low=0, high=1), rng.uniform(low=0, high=1)
#         cmin = -1
#         cmax = +1
#         ra = 0 + u1*(2*np.pi-0)
#         dec = np.pi - (np.arccos(cmin+u2*(cmax-cmin)))
#         ur = np.zeros((nPoints, 3))
#         ur[:, 0] = np.sin(dec) * np.cos(ra)
#         ur[:, 1] = np.sin(dec) * np.sin(ra)
#         ur[:, 2] = np.cos(dec)
#         return ur


# def shuffle(array, seed=None):
#     '''MARCHE PAS'''
#     rng_= mpy.random.MPIRandomState(array.size, seed=seed)
#     idx = rng_.choice(np.arange(0,array.size,1))

#     return array[idx]


# def NFW_mpi(sat_cat, Nb_sat, NFW, rng, seed=None):

#     """
#         --- Compute NFW postion and velocity shifts for satelittes galaxies non multithread method (used for fitting)
#     """

#     if len(NFW) > Nb_sat:
#         np.random.shuffle(NFW)
#         eta = NFW[:Nb_sat]
#     else:
#         eta = NFW[rng.randint(0, len(NFW))]

#     a = 0
#     while len(eta[eta > sat_cat['c']]) > 1:
#         temp = len(eta[eta > sat_cat['c']])
#         if a == temp:
#             break
#         a = temp
#         rng_= mpy.random.MPIRandomState(a, seed=seed)
#         eta[eta > sat_cat['c']] = NFW[rng_.randint(
#             0, high=len(NFW))]
    
#     tet = np.zeros(len(eta[eta > sat_cat['c']]))
#     for i in range(len(eta[eta > sat_cat['c']])):
#         a = eta[sat_cat['c'][eta > sat_cat['c']][i] > eta]
#         tet[i] = a[np.random.randint(len(a))]
#     eta[eta > sat_cat['c']] = tet
#     del tet

#     etaVir = eta/sat_cat['c']  # =r/rvir

#     def f(x):
#         return np.log(1.+x)-x/(1.+x)
#     G = 4.302e-6  # in kpc/Msol (km.s)^2
#     vVir = np.sqrt(G*sat_cat["Mh"]/sat_cat["Rh"])
#     v = vVir * np.sqrt(f(sat_cat['c'] * etaVir)
#                        / (etaVir * f(sat_cat['c'])))

    
#     ur = getPointsOnSphere_mpi(sat_cat.size, rng)
#     uv = getPointsOnSphere_mpi(sat_cat.size, rng)
#     return _rescale(ur, (etaVir*sat_cat["Rh"])), _rescale(uv, v)

