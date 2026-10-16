"""Velocity-error samplers selected by name in a tracer's ``vsmear`` dictionary.

Add a function here and select it with ``model: function_name``. Samplers take
``size``, a velocity scale ``dv`` in km/s, and the supplied NumPy ``rng``. They
return one finite velocity offset per galaxy, with shape ``(size,)``. Optional
entries in ``vsmear.params`` are forwarded as keyword arguments.
Table-based samplers can set ``requires_dv = False`` to allow an omitted scale.
"""


def gaussian(size, dv, rng, **kwargs):
    """Draw zero-centered Gaussian errors with standard deviation ``dv`` [km/s]."""
    return rng.normal(0.0, dv, size=size)


def lorentzian(size, dv, rng, **kwargs):
    """Draw zero-centered Cauchy errors with HWHM ``dv`` [km/s], FWHM ``2*dv``.

    This distribution has no finite mean or variance; its location is zero.
    """
    return dv * rng.standard_cauchy(size=size)


def desi(size, dv, rng, *, tracer, zmin, zmax,
         data_dir='/global/cfs/projectdirs/desi/users/jiaxiyu/repeated_observations/EDR_vs_Y3/LSS-scripts_repeats',
         dvmode='obs', verbose=False):
    """Sample prepared DESI Y3 velocity-error CDFs in km/s.

    This follows ``Y3_redshift_systematics.vsmear``: cubic inverse interpolation
    of ``vbin = log10(abs(delta_v))`` and the normalized ``cdf``, followed by
    paired positive/negative draws and shuffling. Odd sizes drop one shuffled
    draw. The supplied RNG is used without resetting NumPy's global state.

    Select ``model: desi`` and put ``tracer``, ``zmin``, ``zmax``, and
    ``data_dir`` in ``vsmear.params``. Omit ``dv`` or set it to null: the table
    supplies the velocity scale. ``dvmode`` selects ``obs`` or ``model`` tables.
    Files are named ``{tracer}_z{zmin:.1f}-{zmax:.1f}_CDF.npz`` for observations,
    with ``_kernel0.3`` before ``_CDF`` for the model. Tracer subsamples use the
    first three characters, except full names containing ``slitless``.
    """
    from pathlib import Path
    import numpy as np
    from scipy.interpolate import interp1d

    if dv is not None:
        raise ValueError('DESI velocity widths come from the CDF table; omit dv or set it to None')
    if not isinstance(tracer, str) or tracer[:3] not in ('BGS', 'LRG', 'ELG', 'QSO'):
        raise ValueError('DESI tracer must start with BGS, LRG, ELG, or QSO')
    if not np.isfinite(zmin) or not np.isfinite(zmax) or not 0 <= zmin < zmax:
        raise ValueError('DESI redshift limits must be finite and satisfy 0 <= zmin < zmax')
    if dvmode not in ('obs', 'model'):
        raise ValueError('DESI dvmode must be "obs" or "model"')
    if not isinstance(size, (int, np.integer)) or size < 0:
        raise ValueError('DESI sample size must be a nonnegative integer')
    if size == 0:
        return np.empty(0, dtype=np.float32)

    tracer_key = tracer if 'slitless' in tracer else tracer[:3]
    kernel = '' if dvmode == 'obs' else '_kernel0.3'
    filename = Path(data_dir) / f'{tracer_key}_z{zmin:.1f}-{zmax:.1f}{kernel}_CDF.npz'
    if not filename.is_file():
        raise FileNotFoundError(f'DESI velocity-error CDF not found: {filename}. '
                                'Set params.data_dir to the directory containing the prepared Y3 tables.')
    if verbose:
        print(f'Load {filename} to add DESI redshift uncertainties and catastrophics')
    with np.load(filename, allow_pickle=False) as data:
        log_velocity = data['vbin']
        cdf_raw = data['cdf']
    if (log_velocity.ndim != 1 or cdf_raw.shape != log_velocity.shape
            or not np.all(np.isfinite(log_velocity)) or not np.all(np.isfinite(cdf_raw))
            or np.any(np.diff(log_velocity) <= 0) or np.any(np.diff(cdf_raw) < 0)
            or np.any(cdf_raw < 0)):
        raise ValueError(f'Invalid DESI CDF in {filename}: expected finite, increasing vbin '
                         'and a nonnegative, nondecreasing cdf of the same length')
    cdf, indices = np.unique(cdf_raw, return_index=True)
    if cdf.size < 4 or cdf[-1] <= 0:
        raise ValueError(f'Invalid DESI CDF in {filename}: cubic interpolation '
                         'requires at least four distinct CDF values and a positive total')
    grid = log_velocity[indices]
    inverse_cdf = interp1d(cdf / cdf[-1], grid, kind='cubic', bounds_error=False,
                          fill_value=(grid[0], grid[-1]))
    magnitudes = 10 ** inverse_cdf(rng.uniform(0., 1., (size + 1) // 2))
    offsets = np.concatenate((magnitudes, -magnitudes))
    rng.shuffle(offsets)
    if size % 2:
        offsets = offsets[1:]
    return offsets.astype(np.float32)


desi.requires_dv = False
