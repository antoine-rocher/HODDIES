import numpy as np
try:
    from pycorr import TwoPointCorrelationFunction

except ImportError:
    import warnings
    warnings.warn(
        'Could not import pycorr. Install pycorr with ' \
        '"python -m pip install git+https://github.com/cosmodesi/pycorr#egg=pycorr[corrfunc]".' \
        'pycorr currently use a branch of Corrfunc, uninstall previous Corrfunc version (if any): "pip uninstall Corrfunc"'\
        '' 
    )

from scipy.interpolate import interp1d
from scipy.integrate import quad
from numba import njit, prange


def get_list_stat():
    '''
    Return a dictionary mapping clustering statistic names to their corresponding computation functions.
    The supported statistics are:
    - 'xi_smu': 2D correlation function in (s, mu) space.
    - 'xi_rppi': 2D correlation function in (r_p, pi) space.
    - 'delta_sigma': Excess surface density.
    - 'CIC': Count in Cells.
    - 'power_spectrum': Power spectrum.
    '''
    LIST_STAT = ['wp', 'xi_ells', 'xi_smu', 'xi_rppi', 'delta_sigma', 'CIC', 'power_spectrum']

    return LIST_STAT
    


def compute_twopoint(pos1, mode, edges, boxsize=None, los='z', nthreads=32, R1R2=None, pos2=None, **kwargs):
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

    result = TwoPointCorrelationFunction(mode, edges, data_positions1=pos1, data_positions2=pos2, engine='corrfunc', boxsize=boxsize, los=los, nthreads=nthreads, R1R2=R1R2, **kwargs)
    return result



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
#     rpbins = np.geomspace(0.001, rbins.max()*1.2, 100)
#     pibins = np.linspace(-pimax, pimax, 2 * pimax + 1)
#     rp_centres = 0.5 * (rpbins[1:] + rpbins[:-1])
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
#     if np.any(np.isnan(rp)):
#         rp[np.isnan(rp)] = rp_centres[np.isnan(rp)]
#     mask = ~np.isnan(wp)


#     # --- 4) Sigma(R) = rho_m * w_p(R)
#     Sigma = rho_m *wp[mask]

#     # --- 5) Compute Σ(<R)
#     spline_S = interp1d(rp[mask], Sigma, kind="cubic",
#                         bounds_error=False, fill_value="extrapolate")

#     def integrand(r):
#         return r * spline_S(r)
    
#     Sigma_mean = np.array([2.0 / R**2 * quad(integrand, 0, R, limit=200)[0] for R in rp])

#     # --- 6) Excess surface density
#     DeltaSigma = (Sigma_mean - Sigma) /1e12 
#     spline_Dsigma = interp1d(rp, DeltaSigma) 
#     rp_cent = 0.5 * (rbins[1:] + rbins[:-1])

#     return rp_cent, spline_Dsigma(rp_cent)



def compute_delta_sigma(gal_pos, part_pos, boxsize, rbins, rho_m,
                       pimax=None, dpi=1., los='z', nthreads=64,
                       rp_min_int=1e-3, n_int=200):
    """
    Compute excess surface density ΔΣ(R) in units of 1e12[Msun/h / (Mpc/h)^2] via the galaxy-matter cross-correlation.
    
    Sigma_bar(<R) requires the enclosed mass from r = 0, so first wp is measured with a
    fine binning extending well inside the first output bin, with an
    analytic power-law continuation below it. Truncating the cumulative
    integral at the first bin instead makes DeltaSigma go negative at small rp.

    Parameters
    ----------
    gal_pos, part_pos : (N, 3) arrays
        Positions in Mpc/h. Assumed already wrapped into [0, boxsize).
    rbins : array
        Output projected bin edges, Mpc/h.
    rho_m : float
        Comoving mean matter density, (Msun/h) / (Mpc/h)^3.
    pimax : float, optional
        LOS half-depth. Default boxsize/2 (full projection, matches halotools).
    dpi : float
        LOS bin width, Mpc/h.
    rp_min_int : float
        Inner edge of the internal rp grid. Should sit at or above the
        simulation softening; everything below is handled analytically.
    n_int : int
        Number of internal rp bins.

    Returns
    -------
    rp, delta_sigma
        rp and delta_sigma on the output binning.
    """
    if pimax is None:
        pimax = boxsize / 2.

    npi      = int(round(2 * pimax / dpi))
    pi_edges = np.linspace(-pimax, pimax, npi + 1)

    # Internal grid: finer than the output binning and extended inward
    rp_int = np.geomspace(rp_min_int, 1.2 * rbins[-1], n_int + 1)

    result = TwoPointCorrelationFunction(
        'rppi', edges=[rp_int, pi_edges],
        data_positions1=gal_pos.T,
        data_positions2=part_pos.T,
        boxsize=boxsize, los=los,
        position_type='xyz',
        engine='corrfunc', nthreads=nthreads)

    wp = np.nan_to_num(result(pimax=pimax))     # integrated over pi, Mpc/h

    rp_c = np.sqrt(rp_int[:-1] * rp_int[1:])

    # Enclosed contribution from r < rp_c[0], assuming wp ~ r^slope there:
    #   int_0^r0 r' wp(r') dr' = wp(r0) r0^2 / (2 + slope)
    slope = np.log(wp[1:] / wp[:-1]) / np.log(rp_c[1:] / rp_c[:-1])
    inner = wp[0] * rp_c[0]**2 / (2. + slope[~np.isnan(slope)][0])  # Use the first non-NaN slope value

    # integrand = rp_c * wp * np.diff(rp_int)
    # cum       = inner + np.concatenate([[0.], np.cumsum(integrand)])
    # wp_bar    = np.interp(rp_c, rp_int[1:], 2. * cum[1:] / rp_int[1:]**2)

    # int r wp dr = int r^2 wp dln(r), trapezoid in ln r
    lnr  = np.log(rp_c)
    f    = rp_c**2 * wp
    seg  = 0.5 * (f[1:] + f[:-1]) * np.diff(lnr)
    cum  = inner + np.concatenate([[0.], np.cumsum(seg)])
    wp_bar = np.interp(rp_c, rp_c[1:], 2. * cum[1:] / rp_c[1:]**2)

    ds_int = rho_m * (wp_bar - wp) / 1e12       # -> h Msun / pc^2

    rp = np.sqrt(rbins[:-1] * rbins[1:])
    return rp, np.interp(rp, rp_c, ds_int) 


def compute_power_spectrum(pos1, boxsize, kedges, pos2=None, los='z', nmesh=256, resampler='tsc', interlacing=2, ells=(0, 2), **kwargs):
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
        data_positions1=pos1,
        data_positions2=pos2,
        boxsize=boxsize, nmesh=nmesh, edges=kedges,
        los=los, resampler=resampler,
        interlacing=interlacing, ells=ells,
        **kwargs
    )
    
    return result



@njit(fastmath=True, cache=True)
def build_linked_list(X, Y, Z, boxsize, cell_size):

    """
    Build a linked list for efficient neighbor searching in a periodic box.
    """

    N = len(X)
    ncell = int(boxsize / cell_size)

    head = -1 * np.ones((ncell, ncell, ncell), dtype=np.int64)
    linked = -1 * np.ones(N, dtype=np.int64)

    for i in range(N):
        ix = int(X[i] / cell_size) % ncell
        iy = int(Y[i] / cell_size) % ncell
        iz = int(Z[i] / cell_size) % ncell

        linked[i] = head[ix, iy, iz]
        head[ix, iy, iz] = i

    return head, linked, ncell


@njit(fastmath=True, parallel=True, cache=True)
def count_in_cylinder(X, Y, Z, boxsize,
                          R_max, L_max,
                          head, linked, ncell, cell_size):
    """
    Compute count in cylinder.
    
    Parameters
    ----------
    TDB
    """

    N = len(X)
    counts = np.zeros(N, dtype=np.int32)
    R2 = R_max * R_max

    for i in prange(N):
        xi, yi, zi = X[i], Y[i], Z[i]

        ix = int(xi / cell_size) % ncell
        iy = int(yi / cell_size) % ncell
        iz = int(zi / cell_size) % ncell

        c = 0
        dz_cells = int(L_max / cell_size) + 1
        # loop over neighbor cells in XY only
        for dx_cell in (-1, 0, 1):
            for dy_cell in (-1, 0, 1):

                jx = (ix + dx_cell) % ncell
                jy = (iy + dy_cell) % ncell

                # loop over ALL z cells (needed for pi cut)

                for dz_cell in range(-dz_cells, dz_cells + 1):
                    jz = (iz + dz_cell) % ncell

                    j = head[jx, jy, jz]

                    while j != -1:

                        if j != i:
                            dx = X[j] - xi
                            dy = Y[j] - yi
                            dz = Z[j] - zi

                            # periodic wrapping
                            dx -= boxsize * np.round(dx / boxsize)
                            dy -= boxsize * np.round(dy / boxsize)
                            dz -= boxsize * np.round(dz / boxsize)

                            rp2 = dx*dx + dy*dy

                            if rp2 <= R2 and abs(dz) <= L_max:
                                c += 1

                        j = linked[j]

        counts[i] = c

    return counts


@njit(fastmath=True, cache=True )
def compute_CIC(X, Y, Z, boxsize, R_max, L_max):
    """
    Numba grid-based counts-in-cylinder.
    """

    # optimal cell size ~ R_max
    cell_size = R_max

    head, linked, ncell = build_linked_list(
        X, Y, Z, boxsize, cell_size
    )

    return count_in_cylinder(
        X, Y, Z, boxsize,
        R_max, L_max,
        head, linked, ncell, cell_size
    )   