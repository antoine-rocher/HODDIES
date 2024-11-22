import numpy as np
from pycorr import TwoPointCorrelationFunction, project_to_multipoles, project_to_wp
from numba import njit, numba
import os
import HOD_models
import matplotlib.pyplot as plt
import mpytools as mpy
import scipy


def apply_rsd(cat, z, boxsize, H_0=100, los='z', vsmear=0, cosmo=None):
    if cosmo is None :
        from cosmoprimo.fiducial import DESI
        cosmo = DESI(engine='class')
    rsd_factor = 1 / (1 / (1 + z) * H_0 * cosmo.efunc(z))
    pos_rsd = [cat[p] % boxsize if p !=los else (cat[p] + (cat['v'+p] + np.random.normal(0,vsmear, size=len(cat[p])))*rsd_factor) %boxsize if vsmear is not None else (cat[p] + cat['v'+p]*rsd_factor) %boxsize for p in 'xyz']
    return pos_rsd
    
    
def compute_2PCF(pos1, edges, ells=(0, 2), boxsize=None, los='z', nthreads=8, R1R2=None, pos2=None, mpicomm=None):
    """
    --- Compute 2D correlation function and return multipoles for a galaxy/halo catalog in a cubic box.
    """       
        
    result = TwoPointCorrelationFunction('smu', edges, 
                                         data_positions1=pos1, data_positions2=pos2, engine='corrfunc', 
                                         boxsize=boxsize, los=los, nthreads=nthreads, R1R2=R1R2, mpicomm=mpicomm)
    
    return project_to_multipoles(result, ells=ells)

    
def compute_wp(pos1, edges, pimax=40, boxsize=None, los='z', nthreads=8, R1R2=None, pos2=None, mpicomm=None):
    """
    --- Compute 2D correlation function and return multipoles for a galaxy/halo catalog in a cubic box.
    """       
        
    result = TwoPointCorrelationFunction('rppi', edges, 
                                         data_positions1=pos1, data_positions2=pos2, engine='corrfunc', 
                                         boxsize=boxsize, los=los, nthreads=nthreads, R1R2=R1R2, mpicomm=mpicomm)
    
    return project_to_wp(result, pimax=pimax)


@njit(parallel=True, fastmath=True)
def compute_N(log10_Mh, fun_cHOD, fun_sHOD, p_cen, p_sat, p_ab, Nthread, ab_arr=None, conformity=False, seed=None):
    """
    --- Compute the probability N for central galaxies given a HOD model
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
    '''
    --- Aiding function for NFW computation
    '''
    return np.log(1.+x)-x/(1.+x)


@njit(fastmath=True)
def _rescale(a, b):
    '''
    --- Aiding function for NFW computation
    '''
    return np.transpose(np.multiply(np.transpose(a), np.transpose(b)))

@njit(parallel=True, fastmath=True)
def getPointsOnSphere_jit(nPoints, Nthread, seed=None):
    '''
    --- Aiding function for NFW computation, generate random points in a sphere
    '''
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



def rd_draw_NFW(nPoints):
    """
    --- Function to generate random points in a NFW profile using Metropolis-Hastings algorithm 
    """
    
    if nPoints <= 100000:
        raise ValueError ('Error : NPoints must be above 10000')
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
    dataPruned = data[100000:]
    np.random.shuffle(dataPruned)
    os.makedirs('data', exist_ok=True)
    np.save('data/nfw.npy', dataPruned)
    return dataPruned



@njit(parallel=True, fastmath=True)
def compute_ngal(log10_Mh, fun_cHOD, fun_sHOD, Nthread, p_cen, p_sat=None, conformity=False):
    """
    --- Compute the number of galaxy and the satelitte fraction form HOD parameters 
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
    

xt = np.linspace(-1,0,1000000)
ft = np.real(scipy.special.lambertw(xt, k=0))
interp_lambertw= njit(parallel=True, fastmath=True)(lambda x: np.interp(x, xt, ft))

@njit(fastmath=True)
def get_etavir_nfw(c): 
    '''
        Adaptation of approxiamte 3D NFW random sampling from 1805.09550  
    '''
    rd_u = np.random.uniform() * (np.log(1.0 + c)-c/(1.0 + c))
    return (-(1.0/interp_lambertw(-np.exp(-rd_u-1)))-1)/c

@njit(parallel=True, fastmath=True)
def compute_fast_NFW(x_h, y_h, z_h, vx_h, vy_h, vz_h, c, M, Rvir, rd_pos, rd_vel, exp_frac=0, exp_scale=1, nfw_rescale=1, vrms_h=None, f_sigv=None, v_infall=None, vel_sat='NFW', Nthread=64, seed=None):
    
    """
    --- Compute NFW positions and velocities for satelitte galaxies
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


@njit(parallel=True, fastmath=True)
def compute_sat_from_part(xp, yp, zp, vxp, vyp, vzp, 
                          x_sat, y_sat, z_sat, vx_sat, vy_sat, vz_sat, 
                          npout, npstart, nb_sat, cum_sum_sat, Nthread, seed=None):
    
    """
    --- Compute  positions and velocities for satelitte galaxies
    """
    numba.set_num_threads(Nthread)
    mask_nfw = np.zeros(nb_sat.sum(), dtype='bool')
    hstart = np.rint(np.linspace(0, npout.size, Nthread + 1))
    for tid in numba.prange(Nthread):
        if seed is not None:
            np.random.seed(seed[tid])

        for i in range(int(hstart[tid]), int(hstart[tid + 1])):
            if nb_sat[i] < npout[i]:
                tt = np.random.choice(npout[i], nb_sat[i], replace=False) + npstart[i]
                x_sat[cum_sum_sat[i]: cum_sum_sat[i+1]] = xp[tt]
                y_sat[cum_sum_sat[i]: cum_sum_sat[i+1]] = yp[tt]
                z_sat[cum_sum_sat[i]: cum_sum_sat[i+1]] = zp[tt]
                vx_sat[cum_sum_sat[i]: cum_sum_sat[i+1]] = vxp[tt]
                vy_sat[cum_sum_sat[i]: cum_sum_sat[i+1]] = vyp[tt]
                vz_sat[cum_sum_sat[i]: cum_sum_sat[i+1]] = vzp[tt]
                #id_parts[cum_sum_sat[i]: cum_sum_sat[i+1]] = tt + npstart[i]            
            else:
                if npout[i] > 0:
                    tt = np.arange(npout[i]) + npstart[i]
                    x_sat[cum_sum_sat[i]: cum_sum_sat[i] + npout[i]] = xp[tt]
                    y_sat[cum_sum_sat[i]: cum_sum_sat[i] + npout[i]] = yp[tt]
                    z_sat[cum_sum_sat[i]: cum_sum_sat[i] + npout[i]] = zp[tt]
                    vx_sat[cum_sum_sat[i]: cum_sum_sat[i] + npout[i]] = vxp[tt]
                    vy_sat[cum_sum_sat[i]: cum_sum_sat[i] + npout[i]] = vyp[tt]
                    vz_sat[cum_sum_sat[i]: cum_sum_sat[i] + npout[i]] = vzp[tt]
                
                mask_nfw[cum_sum_sat[i]+npout[i]: cum_sum_sat[i+1]] = True
    return mask_nfw



def plot_HOD(p_cen, p_sat, fun_cHOD, fun_sHOD, logM = np.linspace(10.8,15,100), fig=None, color=None, label=None, figsize=(5,4), show=True):
    
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




#### MPI FUNCTIONS


def compute_ngal_mpi(log10_Mh, fun_cHOD, fun_sHOD, p_cen, p_sat=None):
    """
    --- Compute the number of galaxy and the satelitte fraction form HOD parameters
    """
    
    nbinsM, logbinM = np.histogram(log10_Mh, bins=100, range=(10, 16))[:2]
    dM = np.diff(logbinM)[0]
    LogM = 0.5*(logbinM[:-1]+logbinM[1:])

    Ncent = fun_cHOD(LogM, p_cen)
    Ncent[Ncent < 0] = 0
    ngal_c = (nbinsM/dM * Ncent*dM).sum()

    if p_sat is not None:
        M_0 = p_sat[1]
        Nsat = np.zeros(100)
        Nsat[LogM > M_0] = fun_sHOD(LogM[LogM > M_0], p_sat)
        ngal_sat = (nbinsM/dM * Nsat*dM).sum()
    else:
        ngal_sat = 0
    return ngal_c, ngal_sat


def compute_N_mpi(log10_Mh, fun_cHOD, fun_sHOD, p_cen, p_sat=None, p_ab=None, ab_arr=None):

    """
    --- Compute the probability N for central galaxies given a HOD model
    """
    
    Ncent = fun_cHOD(log10_Mh, p_cen)
    if p_ab is not None:
        Ncent *= (1 + np.sum(p_ab[0] * ab_arr)*(1-Ncent))
    if p_sat is None:
        return Ncent, 0
    M_0 = p_sat[1]
    N_sat = np.zeros_like(log10_Mh)
    N_sat[log10_Mh > M_0] = fun_sHOD(log10_Mh[log10_Mh > M_0], p_sat)
    if p_ab is not None:
        N_sat *= (1 + np.sum(p_ab[1] * ab_arr)*(1-N_sat))
    return Ncent, N_sat




def getPointsOnSphere_mpi(nPoints, rng):
        u1, u2 = rng.uniform(low=0, high=1), rng.uniform(low=0, high=1)
        cmin = -1
        cmax = +1
        ra = 0 + u1*(2*np.pi-0)
        dec = np.pi - (np.arccos(cmin+u2*(cmax-cmin)))
        ur = np.zeros((nPoints, 3))
        ur[:, 0] = np.sin(dec) * np.cos(ra)
        ur[:, 1] = np.sin(dec) * np.sin(ra)
        ur[:, 2] = np.cos(dec)
        return ur


def shuffle(array, seed=None):
    '''MARCHE PAS'''
    rng_= mpy.random.MPIRandomState(array.size, seed=seed)
    idx = rng_.choice(np.arange(0,array.size,1))

    return array[idx]


def NFW_mpi(sat_cat, Nb_sat, NFW, rng, seed=None):

    """
        --- Compute NFW postion and velocity shifts for satelittes galaxies non multithread method (used for fitting)
    """

    if len(NFW) > Nb_sat:
        np.random.shuffle(NFW)
        eta = NFW[:Nb_sat]
    else:
        eta = NFW[rng.randint(0, len(NFW))]

    a = 0
    while len(eta[eta > sat_cat['c']]) > 1:
        temp = len(eta[eta > sat_cat['c']])
        if a == temp:
            break
        a = temp
        rng_= mpy.random.MPIRandomState(a, seed=seed)
        eta[eta > sat_cat['c']] = NFW[rng_.randint(
            0, high=len(NFW))]
    
    tet = np.zeros(len(eta[eta > sat_cat['c']]))
    for i in range(len(eta[eta > sat_cat['c']])):
        a = eta[sat_cat['c'][eta > sat_cat['c']][i] > eta]
        tet[i] = a[np.random.randint(len(a))]
    eta[eta > sat_cat['c']] = tet
    del tet

    etaVir = eta/sat_cat['c']  # =r/rvir

    def f(x):
        return np.log(1.+x)-x/(1.+x)
    G = 4.302e-6  # in kpc/Msol (km.s)^2
    vVir = np.sqrt(G*sat_cat["Mh"]/sat_cat["Rh"])
    v = vVir * np.sqrt(f(sat_cat['c'] * etaVir)
                       / (etaVir * f(sat_cat['c'])))

    
    ur = getPointsOnSphere_mpi(sat_cat.size, rng)
    uv = getPointsOnSphere_mpi(sat_cat.size, rng)
    return _rescale(ur, (etaVir*sat_cat["Rh"])), _rescale(uv, v)