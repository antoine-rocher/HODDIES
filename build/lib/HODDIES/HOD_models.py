import numpy as np
import math
from numba import njit, jit


@njit(fastmath=True)
def HMQ(log10_Mh, Ac, Mc, sig_M, gamma, Q, pmax):

    """
    Computes the HMQ (High Mass Quenched) Halo Occupation Distribution (HOD) model 
    without normalization based on arxiv:1910.05095.

    Parameters:
    - log10_Mh: float
        Logarithm (base 10) of halo mass.
    - Ac: float
        Normalization amplitude.
    - Mc: float
        Characteristic halo mass (log10).
    - sig_M: float
        Width of central galaxy mass distribution.
    - gamma: float
        Shape parameter controlling the asymetry.
    - Q: float
        Quenching factor.
    - pmax: float
        Maximum probability.

    Returns:
    - float
        Expected number of galaxies in a halo of mass log10_Mh.
    """

    phi_x = 1 / (np.sqrt(2 * np.pi) * sig_M) * np.exp(
            - (log10_Mh - Mc)**2 / (2 * sig_M**2))
    PHI_gamma_x = 0.5 * (1 + math.erf(gamma * (log10_Mh - Mc) / (sig_M * np.sqrt(2))))
    A = (pmax - 1/Q)
    return Ac * (2 * A * phi_x * PHI_gamma_x + 0.5 / Q * (1 + math.erf((log10_Mh - Mc) / 0.01)))

@njit(fastmath=True)
def mHMQ(log10_Mh, Ac, Mc, sig_M, gamma):

    """
    Computes a simplified version of the HMQ model without normalization and quenching terms 
    based on arxiv:2306.06319.

    Parameters:
    - log10_Mh: float
        Logarithm of halo mass.
    - Ac: float
        Amplitude parameter.
    - Mc: float
        Characteristic halo mass (log10).
    - sig_M: float
        Width of central galaxy mass distribution.
    - gamma: float
        Controls the asymmetry of the distribution.

    Returns:
    - float
        Expected number of galaxies in a halo of mass log10_Mh.
    """

    phi_x = 1 / (np.sqrt(2 * np.pi) * sig_M) * np.exp(
            - (log10_Mh - Mc)**2 / (2 * sig_M**2))
    PHI_gamma_x = 0.5 * (1 + math.erf(gamma * (log10_Mh - Mc) / (sig_M * np.sqrt(2))))
    return Ac * 2 * phi_x * PHI_gamma_x

@njit(fastmath=True)
def GHOD(log10_Mh, Ac, Mc, sig_M):

    """
    Gaussian HOD model based on arXiv:1708.07628.

    Parameters:
    - log10_Mh: float
        Logarithm of halo mass.
    - Ac: float
        Amplitude parameter.
    - Mc: float
        Characteristic halo mass (log10).
    - sig_M: float
        Width of central galaxy mass distribution.
    
    Returns:
    - float
        Expected number of galaxies in a halo of mass log10_Mh.
    """

    return Ac / (np.sqrt(2 * np.pi) * sig_M) * np.exp(-(log10_Mh - Mc)**2 / (2 * sig_M**2))

@njit(fastmath=True)
def LNHOD(log10_Mh, Ac, Mc, sig_M):

    """
    Log-normal HOD model based on arxiv:2306.06319.

    Parameters:
    - log10_Mh: float
        Logarithm of halo mass.
    - Ac: float
        Amplitude parameter.
    - Mc: float
        Characteristic halo mass (log10).
    - sig_M: float
        Width of central galaxy mass distribution.

    Returns:
    - float
        Expected number of galaxies in a halo of mass log10_Mh.
    """

    x = log10_Mh - Mc + 1
    if x <= 0:
        return 0
    val = Ac * np.exp(-(np.log(x))**2 / (2 * sig_M**2)) / (x * sig_M * np.sqrt(2 * np.pi))
    return val

@njit(fastmath=True)
def SFHOD(log10_Mh, Ac, Mc, sig_M, gamma):

    """
    Star-forming HOD model based on arXiv:1708.07628.

    Parameters:
    - log10_Mh: float
        Logarithm of halo mass.
    - Ac: float
        Amplitude parameter.
    - Mc: float
        Characteristic halo mass (log10).
    - sig_M: float
        Width of central galaxy mass distribution.
    - gamma: float
        Controls the asymmetry of the distribution.

    Returns:
    - float
        Expected number of galaxies in a halo of mass log10_Mh.
    """

    norm = Ac / (np.sqrt(2 * np.pi) * sig_M)
    if Mc >= log10_Mh:
        return norm * np.exp(-(log10_Mh - Mc)**2 / (2 * sig_M**2))
    else:
        return norm * (10**log10_Mh / 10**Mc)**gamma

@njit(fastmath=True)
def SHOD(log10_Mh, Ac, Mc, sig_M):

    """
    Standard HOD model (Zheng et al. 2007).

    Parameters:
    - log10_Mh: float
        Logarithm of halo mass.
    - Ac: float
        Amplitude parameter.
    - Mc: float
        Characteristic halo mass (log10).
    - sig_M: float
        Stepness of the step function.

    Returns:
    - float
        Expected number of galaxies in a halo of mass log10_Mh.
    """

    return Ac * 0.5 * (1 + math.erf((log10_Mh - Mc) / sig_M))

@njit(fastmath=True)
def Nsat_pow_law(log10_Mh, As, M_0, M_1, alpha):

    """
    Power-law model for satellite galaxies (Zheng et al. 2005).

    Parameters:
    - log10_Mh: float
    - As: float
    - M_0: float
    - M_1: float
    - alpha: float

    Returns:
    - float
        Expected number of satellite galaxies in a halo of mass log10_Mh.
    """

    N_sat = As * ((10**log10_Mh - 10**M_0) / 10**M_1)**alpha
    return N_sat

@njit(fastmath=True)
def _Nsat_pow_law(log10_Mh, p_sat):
    """
    Wrapper for Nsat_pow_law using parameter array.

    Parameters:
    - log10_Mh: float
    - p_sat: array-like
        Parameters [As, M_0, M_1, alpha]

    Returns:
    - float
        Expected number of satellite galaxies in a halo of mass log10_Mh.
    """
    As, M_0, M_1, alpha = p_sat
    return Nsat_pow_law(log10_Mh, As, M_0, M_1, alpha)

@njit(fastmath=True)
def _SHOD(log10_Mh, p_cen):
    """
    Wrapper for SHOD using parameter array.

    Parameters:
    - log10_Mh: float
    - p_cen: array-like
        Parameters [Ac, Mc, sigM]

    Returns:
    - float
        Expected number of galaxies in a halo of mass log10_Mh.
    """
    Ac, Mc, sigM = p_cen
    return SHOD(log10_Mh, Ac, Mc, sigM)

@njit(fastmath=True)
def _GHOD(log10_Mh, p_cen):
    """
    Wrapper for GHOD using parameter array.

    Parameters:
    - log10_Mh: float
    - p_cen: array-like
        Parameters [Ac, Mc, sigM]

    Returns:
    - float
        Expected number of galaxies in a halo of mass log10_Mh.
    """
    Ac, Mc, sigM = p_cen
    return GHOD(log10_Mh, Ac, Mc, sigM)

@njit(fastmath=True)
def _LNHOD(log10_Mh, p_cen):
    """
    Wrapper for LNHOD using parameter array.

    Parameters:
    - log10_Mh: float
    - p_cen: array-like
        Parameters [Ac, Mc, sigM]

    Returns:
    - float
        Expected number of galaxies in a halo of mass log10_Mh.
    """
    Ac, Mc, sigM = p_cen
    return LNHOD(log10_Mh, Ac, Mc, sigM)

@njit(fastmath=True)
def _SFHOD(log10_Mh, p_cen):
    """
    Wrapper for SFHOD using parameter array.

    Parameters:
    - log10_Mh: float
    - p_cen: array-like
        Parameters [Ac, Mc, sigM, gamma]

    Returns:
    - float
        Expected number of galaxies in a halo of mass log10_Mh.
    """
    Ac, Mc, sigM, gamma = p_cen
    return SFHOD(log10_Mh, Ac, Mc, sigM, gamma)

@njit(fastmath=True)
def _HMQ(log10_Mh, p_cen):
    """
    Wrapper for HMQ using parameter array.

    Parameters:
    - log10_Mh: float
    - p_cen: array-like
        Parameters [Ac, Mc, sig_M, gamma, Q, pmax]

    Returns:
    - float
        Expected number of galaxies in a halo of mass log10_Mh.
    """
    Ac, Mc, sig_M, gamma, Q, pmax = p_cen
    return HMQ(log10_Mh, Ac, Mc, sig_M, gamma, Q, pmax)

@njit(fastmath=True)
def _mHMQ(log10_Mh, p_cen):
    """
    Wrapper for mHMQ using parameter array.

    Parameters:
    - log10_Mh: float
    - p_cen: array-like
        Parameters [Ac, Mc, sig_M, gamma]

    Returns:
    - float
        Expected number of galaxies in a halo of mass log10_Mh.
    """
    Ac, Mc, sig_M, gamma = p_cen
    return mHMQ(log10_Mh, Ac, Mc, sig_M, gamma)
