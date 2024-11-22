import numpy as np
import math
from numba import njit, jit



@njit(fastmath=True)
def HMQ(log10_Mh, Ac, Mc, sig_M, gamma, Q, pmax):
    """
    --- HMQ HOD model modify by Sandy without the normalization
    """
    phi_x = 1 / (np.sqrt(2 * np.pi) * sig_M) * np.exp(
            - (log10_Mh - Mc)**2 / (2 * sig_M**2))
    PHI_gamma_x = 0.5 * (1 + math.erf(gamma * (log10_Mh - Mc)
                                   / (sig_M*np.sqrt(2))))
    A = (pmax - 1/Q)
    return Ac * (2 * A * phi_x * PHI_gamma_x + 0.5 / Q * (1 + math.erf(
            (log10_Mh - Mc) / 0.01)))

@njit(fastmath=True)
def mHMQ(log10_Mh, Ac, Mc, sig_M, gamma):
    """
    --- HMQ HOD model modify by Sandy without the normalization
    """
    phi_x = 1 / (np.sqrt(2 * np.pi) * sig_M) * np.exp(
            - (log10_Mh - Mc)**2 / (2 * sig_M**2))
    PHI_gamma_x = 0.5 * (1 + math.erf(gamma * (log10_Mh - Mc)
                                   / (sig_M*np.sqrt(2))))
    
    return Ac * 2 * phi_x * PHI_gamma_x


@njit(fastmath=True)
def GHOD(log10_Mh, Ac, Mc, sig_M):
    """
    --- Gaussian HOD model from arXiv:1708.07628 (V. Gonzalez-Perez 2018)
    """
    return Ac / (np.sqrt(2 * np.pi) * sig_M) * np.exp(
        -(log10_Mh - Mc)**2 / (2 * sig_M**2))


@njit(fastmath=True)
def LNHOD(log10_Mh, Ac, Mc, sig_M):
    """
    --- HOD model with lognormal distribution 
    """
    x = log10_Mh-Mc+1
    if x <= 0:
        return 0
    val = Ac * np.exp(-(np.log(x))**2 / (
          2 * sig_M**2)) / (x * sig_M * np.sqrt(2 * np.pi))
    #val[np.isnan(val)] = 0
    return val


@njit(fastmath=True)
def SFHOD(log10_Mh, Ac, Mc, sig_M, gamma):
    """
    --- Star forming HOD model from arXiv:1708.07628 (V. Gonzalez-Perez 2018)
    """
    norm = Ac / (np.sqrt(2 * np.pi) * sig_M)
    if Mc >= log10_Mh:
        return norm * np.exp(- (log10_Mh-Mc)**2
                             / (2*sig_M**2))
    else:
        return norm * (10**log10_Mh/10**Mc)**gamma


@njit(fastmath=True)
def SHOD(log10_Mh, Ac, Mc, sig_M):
    """
    --- Standard HOD model from arXiv:astro-ph/0408564 Zheng et al. (2007)
    """
    return Ac * 0.5 * (1 + math.erf((log10_Mh-Mc) / (sig_M)))


@njit(fastmath=True)
def Nsat_pow_law(log10_Mh, As, M_0, M_1, alpha):
    """
    ---  Standard Zheng et al. (2005) satellite HOD parametrization arXiv:astro-ph/0408564
    """
    N_sat = As * ((10**log10_Mh - 10**M_0) / 10**M_1)**alpha
    return N_sat


@njit(fastmath=True)
def _Nsat_pow_law(log10_Mh, p_sat):
    As, M_0, M_1, alpha = p_sat
    return Nsat_pow_law(log10_Mh, As, M_0, M_1, alpha)


@njit(fastmath=True)
def _SHOD(log10_Mh, p_cen):
    Ac, Mc, sigM = p_cen
    return SHOD(log10_Mh, Ac, Mc, sigM)

@njit(fastmath=True)
def _GHOD(log10_Mh, p_cen):
    Ac, Mc, sigM = p_cen
    return GHOD(log10_Mh, Ac, Mc, sigM)

@njit(fastmath=True)
def _LNHOD(log10_Mh, p_cen):
    Ac, Mc, sigM = p_cen
    return LNHOD(log10_Mh, Ac, Mc, sigM)

@njit(fastmath=True)
def _SFHOD(log10_Mh, p_cen):
    Ac, Mc, sigM, gamma = p_cen
    return SFHOD(log10_Mh, Ac, Mc, sigM, gamma)

@njit(fastmath=True)
def _HMQ(log10_Mh, p_cen):
    Ac, Mc, sig_M, gamma, Q, pmax = p_cen
    return HMQ(log10_Mh, Ac, Mc, sig_M, gamma, Q, pmax)

@njit(fastmath=True)
def _mHMQ(log10_Mh, p_cen):
    Ac, Mc, sig_M, gamma = p_cen
    return mHMQ(log10_Mh, Ac, Mc, sig_M, gamma)