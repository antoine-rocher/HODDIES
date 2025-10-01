import time
import gc
import numpy as np
import numpy.linalg as la
import numba
from scipy.fft import irfftn, rfftn
from scipy.ndimage import gaussian_filter
import timeit
import warnings

"""
Originally written by Boryana Hadzhiyska for the ancient: https://arxiv.org/abs/1512.03402.
"""



__all__ = ['tsc_parallel', 'partition_parallel']


def tsc_parallel(
    pos,
    densgrid,
    box,
    weights=None,
    nthread=-1,
    wrap=True,
    npartition=None,
    sort=False,
    coord=0,
    verbose=False,
    offset=0.0,
):
    """
    Function adapted from AbacusHOD (https://github.com/abacusorg/abacusutils/blob/main/abacusnbody/analysis/tsc.py) 
    A parallel implementation of TSC mass assignment using numba. The algorithm
    partitions the particles into stripes of sufficient width that their TSC
    clouds don't overlap, and then does all the even-numbered stripes in
    parallel, followed by the odd-numbered ones.

    This method parallelizes well and can exceed, e.g., 500 million particles
    per second with 32 cores on a modern processor for a cache-resident grid
    size. That's 20 ms for 10^7 particles, which is number density 1e-3 in a
    2 Gpc/h box.

    The algorithm is expected to be bound by memory bandwidth, rather than
    CPU.  Sometimes using, e.g., half of all CPUs will be faster than using all
    of them.

    ``npartition`` is a tuning parameter.  Generally it should be at least
    ``2*nthread``, so that all threads have work to do in both passes.  Sometimes
    an even finer partitioning can produce a favorable ordering in memory of
    particles for TSC.  Sorting the particles within each stripe produces an
    even more favorable ordering, but the current implementation of sorting is
    slow enough that it's usually not worth it.

    To process particles in batches, you can allocate a grid ahead of time and
    pass it in as ``densgrid``.  The grid will *not* be zeroed before the TSC,
    so you can accumulate results incrementally.

    Likewise, this routine applies no normalization to the results beyond the
    particle weights, so accumulating results in batches is straightforward.

    Parameters
    ----------
    pos : ndarray of shape (n,3)
        The particles, in domain [0,box)

    densgrid : ndarray, tuple, or int
        Either an ndarray in which to write the density, or a tuple/int
        indicating the shape of the array to allocate. Can be 2D or 3D; ints
        are interpreted as 3D cubic grids. Anisotropic grids are also supported
        (nx != ny != nz).
        If densgrid is an ndarray, the return value will be None.

    box : float
        The domain size. Positions are expected in domain [0,box) (but may be
        wrapped; see ``wrap``).tsc_parallelc wrap to any out-of-bounds particle positions,
        bringing them back to domain [0,box).  This is on by default
        because it's generally fast compared to TSC.
        Default: True

    npartition : int, optional
        Number of stripes in which to partition the positions.  This is a
        tuning parameter (with certain constraints on the max value); a value
        of None will use the default of (no larger than) 2*nthread.
        Default: None

    sort : bool, optional
        Sort the particles along the ``coord`` coordinate within each partition
        stripe. This can affect performance.
        Default: False

    coord : int, optional
        The coordinate on which to partition. ``coord = 0`` means ``x``,
        ``coord = 1`` means ``y``, etc.
        Default: 0

    verbose : bool, optional
        Print some information about settings and timings.
        Default: False

    Returns
    -------
    ndarray or None
        If ``densgrid`` is an ndarray, returns None. Otherwise, returns the newly
        allocated density grid.
    """

    if nthread < 0:
        nthread = numba.config.NUMBA_NUM_THREADS
    if verbose:
        print(f'nthread={nthread}')

    numba.set_num_threads(nthread)
    if isinstance(densgrid, (int, np.integer)):
        densgrid = (densgrid, densgrid, densgrid)
    if isinstance(densgrid, tuple):
        densgrid = _zeros_parallel(densgrid)
        user_supplied_grid = False
    else:
        user_supplied_grid = True
    n1d = densgrid.shape[coord]

    if not npartition:
        if nthread > 1:
            # Can be equal to n1d//2, or less than or equal to n1d//3.
            # Must be even, and need not exceed 2*nthread.
            if 2 * nthread >= n1d // 2:
                npartition = n1d // 2
                npartition = 2 * (npartition // 2)
                if npartition < n1d // 2:
                    npartition = n1d // 3
            else:
                npartition = min(n1d // 3, 2 * nthread)
            npartition = 2 * (npartition // 2)  # must be even
        else:
            npartition = 1

    if npartition > n1d // 3 and npartition != n1d // 2 and nthread > 1:
        raise ValueError(
            f'npartition {npartition} must be less than'
            f' ngrid//3 = {n1d // 3} or equal to ngrid//2 = {n1d // 2}'
        )
    if npartition > 1 and npartition % 2 != 0 and nthread > 1:
        raise ValueError(f'npartition {npartition} not divisible by 2')
    if verbose and nthread > 1 and npartition < 2 * nthread:
        print(
            f'npartition {npartition} not large enough to use'
            f' all {nthread} threads; should be 2*nthread',
            stacklevel=2,
        )

    def _check_dtype(a, name):
        if a.itemsize > 4:
            warnings.warn(
                f'{name}.dtype={a.dtype} instead of np.float32. '
                'float32 is recommended for performance.',
            )

    _check_dtype(pos, 'pos')
    _check_dtype(densgrid, 'densgrid')
    if weights is not None:
        _check_dtype(weights, 'weights')

    if verbose:
        print(f'npartition={npartition}')

    wraptime = -timeit.default_timer()
    if wrap:
        # This could be on-the-fly instead of in-place, if needed
        _wrap_inplace(pos, box)
    wraptime += timeit.default_timer()
    if verbose:
        print(f'Wrap time: {wraptime:.4g} sec')

    if npartition > 1:
        parttime = -timeit.default_timer()
        ppart, starts, wpart = partition_parallel(
            pos,
            npartition,
            box,
            weights=weights,
            nthread=nthread,
            coord=coord,
            sort=sort,
        )
        parttime += timeit.default_timer()
        if verbose:
            print(f'Partition time: {parttime:.4g} sec')
    else:
        ppart = pos
        wpart = weights
        starts = np.array([0, len(pos)], dtype=np.int64)

    tsctime = -timeit.default_timer()
    _tsc_parallel(ppart, starts, densgrid, box, weights=wpart, offset=offset)
    tsctime += timeit.default_timer()

    if verbose:
        print(f'TSC time: {tsctime:.4g} sec')

    if user_supplied_grid:
        return None
    return densgrid


@numba.njit(parallel=True)
def _zeros_parallel(shape, dtype=np.float32):
    arr = np.empty(shape, dtype=dtype)

    for i in numba.prange(shape[0]):
        arr[i] = 0.0

    return arr


@numba.njit(parallel=True)
def _wrap_inplace(pos, box):
    for i in numba.prange(len(pos)):
        for j in range(3):
            if pos[i, j] >= box:
                pos[i, j] -= box
            elif pos[i, j] < 0:
                pos[i, j] += box


@numba.njit(parallel=True)
def _tsc_parallel(ppart, starts, dens, box, weights, offset):
    npartition = len(starts) - 1
    for i in numba.prange((npartition + 1) // 2):
        if weights is not None:
            wslice = weights[starts[2 * i] : starts[2 * i + 1]]
        else:
            wslice = None
        _tsc_scatter(
            ppart[starts[2 * i] : starts[2 * i + 1]],
            dens,
            box,
            weights=wslice,
            offset=offset,
        )
    if npartition > 1:
        for i in numba.prange((npartition + 1) // 2):
            if weights is not None:
                wslice = weights[starts[2 * i + 1] : starts[2 * i + 2]]
            else:
                wslice = None
            _tsc_scatter(
                ppart[starts[2 * i + 1] : starts[2 * i + 2]],
                dens,
                box,
                weights=wslice,
                offset=offset,
            )


@numba.njit(parallel=True, fastmath=True)
def partition_parallel(
    pos,
    npartition,
    boxsize,
    weights=None,
    coord=0,
    nthread=-1,
    sort=False,
):
    """
    A parallel partition.  Partitions a set of positions into ``npartition``
    pieces, using the ``coord`` coordinate (``coord=0`` partitions on ``x``, ``coord=1``
    partitions on ``y``, etc.).

    The particle copy stage is coded as a scatter rather than a gather.

    Note that this function is expected to be bound by memory bandwidth rather
    than CPU.

    Parameters
    ----------
    pos : ndarray of shape (n,3)
        The positions, in domain [0,boxsize)

    npartition : int
        The number of partitions

    boxsize : float
        The domain of the particles

    weights : ndarray of shape (n,), optional
        Particle weights.
        Default: None

    coord : int, optional
        The coordinate to partition on. 0 is x, 1 is y, etc.
        Default: 0 (x coordinate)

    nthread : int, optional
        Number of threads to parallelize over (using Numba threading).
        Values < 0 use ``numba.config.NUMBA_NUM_THREADS``, which is usually all
        CPUs.
        Default: -1

    sort : bool, optional
        Sort the particles on the ``coord`` coordinate within each partition.
        Can speed up subsequent TSC, but generally slow and not worth it.
        Default: False

    Returns
    -------
    partitioned : ndarray like ``pos``
        The particles, in partitioned order

    part_starts : ndarray, shape (npartition + 1,), dtype int64
        The index in ``partitioned`` where each partition starts

    wpart : ndarray or None
        The weights, in partitioned order; or None if ``weights`` not given.
    """

    if nthread < 0:
        nthread = numba.config.NUMBA_NUM_THREADS
    numba.set_num_threads(nthread)

    assert pos.shape[1] == 3

    # First pass: compute key and per-thread histogram
    dtype = pos.dtype.type
    inv_pwidth = dtype(npartition / boxsize)
    keys = np.empty(len(pos), dtype=np.int32)
    counts = np.zeros((nthread, npartition), dtype=np.int32)
    tstart = np.linspace(0, len(pos), nthread + 1).astype(np.int64)
    for t in numba.prange(nthread):
        for i in range(tstart[t], tstart[t + 1]):
            keys[i] = min(np.int32(pos[i, coord] * inv_pwidth), npartition - 1)
            counts[t, keys[i]] += 1

    # Compute start indices for parallel scatter
    pointers = np.empty(nthread * npartition, dtype=np.int64)
    pointers[0] = 0
    pointers[1:] = np.cumsum(counts.T)[:-1]
    pointers = np.ascontiguousarray(pointers.reshape(npartition, nthread).T)

    starts = np.empty(npartition + 1, dtype=np.int64)
    starts[:-1] = pointers[0]
    starts[-1] = len(pos)

    # Do parallel scatter, specializing for weights to help Numba
    psort = np.empty_like(pos)
    if weights is not None:
        wsort = np.empty_like(weights)
        for t in numba.prange(nthread):
            for i in range(tstart[t], tstart[t + 1]):
                k = keys[i]
                s = pointers[t, k]
                for j in range(3):
                    psort[s, j] = pos[i, j]
                wsort[s] = weights[i]
                pointers[t, k] += 1

        if sort:
            for i in numba.prange(npartition):
                part = psort[starts[i] : starts[i + 1]]
                iord = part[:, coord].argsort()
                part[:] = part[iord]
                weightspart = wsort[starts[i] : starts[i + 1]]
                weightspart[:] = weightspart[iord]
    else:
        wsort = None
        for t in numba.prange(nthread):
            for i in range(tstart[t], tstart[t + 1]):
                k = keys[i]
                s = pointers[t, k]
                for j in range(3):
                    psort[s, j] = pos[i, j]
                pointers[t, k] += 1

        if sort:
            for i in numba.prange(npartition):
                part = psort[starts[i] : starts[i + 1]]
                iord = part[:, coord].argsort()
                part[:] = part[iord]

    return psort, starts, wsort


@numba.njit
def _rightwrap(x, L):
    if x >= L:
        return x - L
    return x


@numba.njit(fastmath=True)
def _tsc_scatter(positions, density, boxsize, weights=None, offset=0.0):
    """
    TSC worker function. Expects particles in domain [0,boxsize).
    Supports 3D and 2D.
    """
    ftype = positions.dtype.type
    itype = np.int16
    threeD = density.ndim == 3
    gx = itype(density.shape[0])
    gy = itype(density.shape[1])
    if threeD:
        gz = itype(density.shape[2])

    inv_hx = ftype(gx / boxsize)
    inv_hy = ftype(gy / boxsize)
    if threeD:
        inv_hz = ftype(gz / boxsize)

    offset = ftype(offset)
    W = ftype(1.0)
    have_W = weights is not None

    HALF = ftype(0.5)
    P75 = ftype(0.75)
    for n in range(len(positions)):
        if have_W:
            W = ftype(weights[n])

        # convert to a position in the grid
        px = (positions[n, 0] + offset) * inv_hx
        py = (positions[n, 1] + offset) * inv_hy
        if threeD:
            pz = (positions[n, 2] + offset) * inv_hz

        # round to nearest cell center
        ix = itype(round(px))
        iy = itype(round(py))
        if threeD:
            iz = itype(round(pz))

        # calculate distance to cell center
        dx = ftype(ix) - px
        dy = ftype(iy) - py
        if threeD:
            dz = ftype(iz) - pz

        # find the tsc weights for each dimension
        wx = P75 - dx**2
        wxm1 = HALF * (HALF + dx) ** 2
        wxp1 = HALF * (HALF - dx) ** 2
        wy = P75 - dy**2
        wym1 = HALF * (HALF + dy) ** 2
        wyp1 = HALF * (HALF - dy) ** 2
        if threeD:
            wz = P75 - dz**2
            wzm1 = HALF * (HALF + dz) ** 2
            wzp1 = HALF * (HALF - dz) ** 2
        else:
            wz = ftype(1.0)

        # find the wrapped x,y,z grid locations of the points we need to change
        # negative indices will be automatically wrapped
        ixm1 = _rightwrap(ix - itype(1), gx)
        ixw = _rightwrap(ix, gx)
        ixp1 = _rightwrap(ix + itype(1), gx)
        iym1 = _rightwrap(iy - itype(1), gy)
        iyw = _rightwrap(iy, gy)
        iyp1 = _rightwrap(iy + itype(1), gy)
        if threeD:
            izm1 = _rightwrap(iz - itype(1), gz)
            izw = _rightwrap(iz, gz)
            izp1 = _rightwrap(iz + itype(1), gz)
        else:
            izw = itype(0)

        # change the 9 or 27 cells that the cloud touches
        density[ixm1, iym1, izw] += wxm1 * wym1 * wz * W
        density[ixm1, iyw, izw] += wxm1 * wy * wz * W
        density[ixm1, iyp1, izw] += wxm1 * wyp1 * wz * W
        density[ixw, iym1, izw] += wx * wym1 * wz * W
        density[ixw, iyw, izw] += wx * wy * wz * W
        density[ixw, iyp1, izw] += wx * wyp1 * wz * W
        density[ixp1, iym1, izw] += wxp1 * wym1 * wz * W
        density[ixp1, iyw, izw] += wxp1 * wy * wz * W
        density[ixp1, iyp1, izw] += wxp1 * wyp1 * wz * W

        if threeD:
            density[ixm1, iym1, izm1] += wxm1 * wym1 * wzm1 * W
            density[ixm1, iym1, izp1] += wxm1 * wym1 * wzp1 * W

            density[ixm1, iyw, izm1] += wxm1 * wy * wzm1 * W
            density[ixm1, iyw, izp1] += wxm1 * wy * wzp1 * W

            density[ixm1, iyp1, izm1] += wxm1 * wyp1 * wzm1 * W
            density[ixm1, iyp1, izp1] += wxm1 * wyp1 * wzp1 * W

            density[ixw, iym1, izm1] += wx * wym1 * wzm1 * W
            density[ixw, iym1, izp1] += wx * wym1 * wzp1 * W

            density[ixw, iyw, izm1] += wx * wy * wzm1 * W
            density[ixw, iyw, izp1] += wx * wy * wzp1 * W

            density[ixw, iyp1, izm1] += wx * wyp1 * wzm1 * W
            density[ixw, iyp1, izp1] += wx * wyp1 * wzp1 * W

            density[ixp1, iym1, izm1] += wxp1 * wym1 * wzm1 * W
            density[ixp1, iym1, izp1] += wxp1 * wym1 * wzp1 * W

            density[ixp1, iyw, izm1] += wxp1 * wy * wzm1 * W
            density[ixp1, iyw, izp1] += wxp1 * wy * wzp1 * W

            density[ixp1, iyp1, izm1] += wxp1 * wyp1 * wzm1 * W
            density[ixp1, iyp1, izp1] += wxp1 * wyp1 * wzp1 * W



def smooth_density(D, R, N_dim, Lbox):
    # cell size
    cell = Lbox / N_dim
    # smoothing scale
    R /= cell
    D_smooth = gaussian_filter(D, R)
    return D_smooth


# tophat
@numba.njit
def Wth(ksq, r):
    k = np.sqrt(ksq)
    w = 3 * (np.sin(k * r) - k * r * np.cos(k * r)) / (k * r) ** 3
    return w


# gaussian
@numba.njit
def Wg(k, r):
    return np.exp(-k * r * r / 2.0)


@numba.njit(parallel=False, fastmath=True)  # parallel=True gives seg fault
def get_tidal(dfour, karr, N_dim, R, dtype=np.float32):
    # initialize array
    tfour = np.zeros((N_dim, N_dim, N_dim // 2 + 1, 6), dtype=np.complex64)

    # computing tidal tensor
    for a in range(N_dim):
        for b in range(N_dim):
            for c in numba.prange(N_dim // 2 + 1):
                if a * b * c == 0:
                    continue

                ksq = dtype(karr[a] ** 2 + karr[b] ** 2 + karr[c] ** 2)
                dok2 = dfour[a, b, c] / ksq

                # smoothed density Gauss fourier
                # dksmo[a, b, c] = Wg(ksq)*dfour[a, b, c]
                # smoothed density TH fourier
                # dkth[a, b, c] = Wth(ksq)*dfour[a, b, c]
                # 0,0 is 0; 0,1 is 1; 0,2 is 2; 1,1 is 3; 1,2 is 4; 2,2 is 5
                tfour[a, b, c, 0] = karr[a] * karr[a] * dok2
                tfour[a, b, c, 3] = karr[b] * karr[b] * dok2
                tfour[a, b, c, 5] = karr[c] * karr[c] * dok2
                tfour[a, b, c, 1] = karr[a] * karr[b] * dok2
                tfour[a, b, c, 2] = karr[a] * karr[c] * dok2
                tfour[a, b, c, 4] = karr[b] * karr[c] * dok2
                if R is not None:
                    tfour[a, b, c, :] *= Wth(ksq, R)
    return tfour


@numba.njit(parallel=False, fastmath=True)
def get_shear_nb(tidr, N_dim):
    shear = np.zeros(shape=(N_dim, N_dim, N_dim), dtype=np.float32)
    tensor = np.zeros((3, 3), dtype=np.float32)
    for a in range(N_dim):
        for b in range(N_dim):
            for c in range(N_dim):
                t = tidr[a, b, c, :]
                tensor[0, 0] = t[0]
                tensor[0, 1] = t[1]
                tensor[0, 2] = t[2]
                tensor[1, 0] = t[1]
                tensor[1, 1] = t[3]
                tensor[1, 2] = t[4]
                tensor[2, 0] = t[2]
                tensor[2, 1] = t[4]
                tensor[2, 2] = t[5]
                evals = la.eigvals(tensor)
                l1 = evals[0]
                l2 = evals[1]
                l3 = evals[2]
                shear[a, b, c] = np.sqrt(
                    0.5 * ((l2 - l1) ** 2 + (l3 - l1) ** 2 + (l3 - l2) ** 2)
                )
    return shear


def calc_shear_from_part(pos_parts, Lbox, cell_size=5, R=1.5, workers=-1, dtype=np.float32):

    start = time.time()
    if pos_parts.shape[0]==3:
        pos_parts=pos_parts.T
    N_dim = int(Lbox/cell_size)
    dens = tsc_parallel(pos_parts, N_dim, Lbox)
    print('finished TSC, took time', time.time() - start)
    start = time.time()
    dsmo = smooth_density(dens, R, N_dim, Lbox)
    print('finished smoothing, took time', time.time() - start)
    start = time.time()
    if isinstance(dsmo, str):
        dsmo = np.load(dsmo)

    # fourier transform the density field
    dsmo = dsmo.astype(dtype)
    dfour = rfftn(dsmo, overwrite_x=True, workers=workers)
    del dsmo
    gc.collect()

    # k values
    karr = np.fft.fftfreq(N_dim, d=Lbox / (2 * np.pi * N_dim)).astype(dtype)

    # compute fourier tidal
    start = time.time()
    tfour = get_tidal(dfour, karr, N_dim, R)
    del dfour
    gc.collect()
    print('finished fourier tidal, took time', time.time() - start)

    # compute real tidal
    start = time.time()
    tidr = irfftn(tfour, axes=(0, 1, 2), workers=workers).real
    del tfour
    gc.collect()
    print('finished tidal, took time', time.time() - start)
    # compute shear
    start = time.time()
    shear = get_shear_nb(tidr, N_dim)
    del tidr
    gc.collect()
    print('finished shear, took time', time.time() - start)
    return shear

def calc_env(pos_parts, Lbox, cell_size=5, R=1.5):

    start = time.time()
    if pos_parts.shape[0]==3:
        pos_parts=pos_parts.T
    N_dim = int(Lbox/cell_size)
    dens = tsc_parallel(pos_parts, N_dim, Lbox)
    print('finished TSC, took time', time.time() - start)
    start = time.time()
    dsmo = smooth_density(dens, R, N_dim, Lbox)
    print('finished smoothing, took time', time.time() - start)
    return dsmo


def calc_shear_from_dsmo(dsmo, Lbox, cell_size=5, R=1.5, workers=-1, dtype=np.float32):

    N_dim = int(Lbox/cell_size)
    # fourier transform the density field
    dsmo = dsmo.astype(dtype)
    dfour = rfftn(dsmo, overwrite_x=True, workers=workers)
    del dsmo
    gc.collect()

    # k values
    karr = np.fft.fftfreq(N_dim, d=Lbox / (2 * np.pi * N_dim)).astype(dtype)

    # compute fourier tidal
    start = time.time()
    tfour = get_tidal(dfour, karr, N_dim, R)
    del dfour
    gc.collect()
    print('finished fourier tidal, took time', time.time() - start)

    # compute real tidal
    start = time.time()
    tidr = irfftn(tfour, axes=(0, 1, 2), workers=workers).real
    del tfour
    gc.collect()
    print('finished tidal, took time', time.time() - start)
    # compute shear
    start = time.time()
    shear = get_shear_nb(tidr, N_dim)
    del tidr
    gc.collect()
    print('finished shear, took time', time.time() - start)
    return shear


def compute_env_shear_abacus(sim_name, zsim, cell_size=5, R=1.5, dir_to_save='/global/homes/a/arocher/users_arocher/HODDIES_data/environemental_quantities/Abacus', root_abacus_dir='/dvs_ro/cfs/cdirs/desi/cosmosim/Abacus'):
    import os
    import glob 
    from abacusnbody.data.read_abacus import read_asdf
    from HODDIES.abacus_io import get_boxsize_from_simname
    import hdf5plugin
    import h5py    

    path = os.path.join(root_abacus_dir, sim_name, 'halos', f'z{zsim:.3f}')
    if not os.path.exists(path):
        raise NameError(f'Wrong simulation path: {path}')
    if zsim not in [0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 1.1, 1.4, 1.7, 2.0, 2.5, 3.0]:
        raise NameError(f'Redhsift snapshot {zsim} does not have particles outputs')

    


    fns = glob.glob(os.path.join(path, 'field_rv_A/*asdf')) + glob.glob(os.path.join(path, 'halo_rv_A/*asdf'))
    start = time.time()

    partpos = []
    for efn in fns:
        print(efn)
        ecat = read_asdf(efn, load=['pos'])
        partpos += [ecat['pos']]
    partpos = np.concatenate(partpos)
    print('compiled all particles', len(partpos), 'took time', time.time() - start)

    Lbox = get_boxsize_from_simname(path)
    dsmo = calc_env(partpos, Lbox, cell_size=cell_size, R=R) 
    shear = calc_shear_from_dsmo(dsmo, Lbox, cell_size=cell_size, R=R, workers=-1) 
    
    if dir_to_save is not None:
        path_to_save = os.path.join(dir_to_save, f'env_shear_map_{sim_name}_z{zsim:.3f}.h5')
        print(f'Save to {path_to_save}')
        with h5py.File(path_to_save, "w") as f:
            f.create_dataset(
                'density',
                data=dsmo,
                **hdf5plugin.Blosc(cname="zstd", clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE))
                
            f.create_dataset(
                'shear',
                data=shear,
                **hdf5plugin.Blosc(cname="zstd", clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE))
    return dsmo, shear