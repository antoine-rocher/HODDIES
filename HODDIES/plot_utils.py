
"""
Generic plotting of clustering statistics.

One statistic per panel. Which statistics are shown is given as input, and
new statistics are added by registering a `StatSpec` -- no change to the
plotting code itself.

If measured data (+ errors) are supplied for a statistic, the data are shown
as points with error bars and a residual sub-panel (model - data)/sigma is
added underneath that panel.
"""

import warnings
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence, Mapping
import numpy as np



# ----------------------------------------------------------------------
# Statistic registry
# ----------------------------------------------------------------------
@dataclass
class StatSpec:
    """Declarative description of one statistic.

    Parameters
    ----------
    source : str
        Key under which this quantity appears in the dict returned by
        ``obj.compute_stats(cat, stat=[...])``, i.e.
        ``tt[source][tracer]``. Several statistics may share one source
        (e.g. the multipoles all come from ``'xi_ells'``).
    xlabel, ylabel : str
        Axis labels (LaTeX allowed).
    component : int or None
        Index into ``y`` when the source holds several components.
        ``None`` means ``y`` is already the quantity to plot.
    scale : callable or None
        ``(x, y) -> y_plot``, applied before plotting. Use for the usual
        ``x * y`` presentation. ``None`` means plot ``y`` as is.
    xscale, yscale : str
        Matplotlib axis scales.
    ndim : int
        1 for curves, 2 for maps drawn with ``pcolormesh``.
    mirror : bool
        (2D only) Mirror the map into all four quadrants, i.e. reflect
        about both axes, giving the usual wedge plot of
        :math:`\\xi(r_p, \\pi)`.
    linthresh : float or None
        (2D only) If set, use a ``symlog`` x-axis with this linear
        threshold -- needed when mirroring a logarithmically binned
        coordinate through zero.
    levels : sequence or None
        (2D only) Contour levels overlaid on the map. ``None`` for no
        contours.
    shading : str
        (2D only) ``pcolormesh`` shading; ``'gouraud'`` interpolates
        across cells for an ``imshow``-like smooth map.
    prepare : callable or None
        ``entry -> (x, y)``, applied to the raw ``tt[source][tracer]``
        value before plotting. Use when the stored result is not already
        a coordinate/value pair -- e.g. CIC returns per-object neighbour
        counts that must be histogrammed first.
    residual : {'sigma', 'ratio'}
        How the residual sub-panel is computed. ``'sigma'`` plots
        ``(model - data)/sigma``; ``'ratio'`` plots ``model/data - 1``,
        which is the natural comparison for a counts distribution.
    residual_ylim : tuple or None
        Explicit y-limits for the residual panel.
    getter_kwargs : dict
        Extra keyword arguments forwarded to ``compute_stats``.
    """
    source: str
    xlabel: str = ''
    ylabel: str = ''
    component: Optional[int] = None
    scale: Optional[Callable] = None
    xscale: str = 'log'
    yscale: str = 'linear'
    ndim: int = 1
    mirror: bool = False
    linthresh: Optional[float] = None
    levels: Optional[Sequence] = None
    shading: str = 'gouraud'
    prepare: Optional[Callable] = None
    residual: str = 'sigma'
    residual_ylim: Optional[tuple] = None
    getter_kwargs: dict = field(default_factory=dict)


def _cic_hist(entry, bins=None, density=True):
    """Histogram raw counts-in-cells values into (centres, P(N)).

    ``entry`` is the array of per-object neighbour counts returned by
    ``compute_stats``; if it is already a ``(centres, values)`` pair it is
    passed through unchanged.
    """
    if isinstance(entry, (tuple, list)) and len(entry) == 2:
        return np.asarray(entry[0]), np.asarray(entry[1])
    counts = np.asarray(entry).ravel()
    counts = counts[np.isfinite(counts)]
    if bins is None:
        hi = int(np.nanmax(counts)) + 1 if counts.size else 15
        bins = np.arange(-0.5, hi + 1.5)
    hist, edges = np.histogram(counts, bins=bins)
    cens = 0.5 * (edges[:-1] + edges[1:])
    if density and hist.sum() > 0:
        hist = hist / hist.sum()
    return cens, hist


_MPC = r'[$\mathrm{Mpc}/h$]'

#: Built-in statistics. Extend with :func:`register_stat`.

STATS = {
    'wp': StatSpec(
        source='wp',
        xlabel=r'$r_p$ ' + _MPC,
        ylabel=r'$r_p \cdot w_p(r_p)$ ' + r'[$(\mathrm{Mpc}/h)^2$]',
        scale=lambda x, y: x * y),

    # Multipoles. These defaults assume get_xiells returns the orders in
    # the order [0, 2, 4]; requesting the group 'xi_ells' instead re-reads
    # the configured multipole list and overrides these with the correct
    # component indices.
    'xi0': StatSpec(
        source='xi_ells', component=0,
        xlabel=r'$s$ ' + _MPC,
        ylabel=r'$s \cdot \xi_0(s)$ ' + _MPC,
        scale=lambda x, y: x * y),

    'xi2': StatSpec(
        source='xi_ells', component=1,
        xlabel=r'$s$ ' + _MPC,
        ylabel=r'$s \cdot \xi_2(s)$ ' + _MPC,
        scale=lambda x, y: x * y),

    'xi4': StatSpec(
        source='xi_ells', component=2,
        xlabel=r'$s$ ' + _MPC,
        ylabel=r'$s \cdot \xi_4(s)$ ' + _MPC,
        scale=lambda x, y: x * y),

    'CIC': StatSpec(
        source='CIC', prepare=lambda e: _cic_hist(e),
        xlabel=r'$N_\mathrm{CIC}$',
        ylabel=r'$P(N_\mathrm{CIC})$',
        xscale='linear', yscale='log',
        residual='ratio', residual_ylim=(-0.3, 0.3)),

    'delta_sigma': StatSpec(
        source='delta_sigma',
        xlabel=r'$R$ ' + _MPC,
        ylabel=r'$R \cdot \Delta\Sigma(R)$ ' + r'[$M_\odot/\mathrm{pc}$]',
        scale=lambda x, y: x * y),

    # 2D statistics
    'xi_rppi': StatSpec(
        source='xi_rppi', ndim=2,
        xlabel=r'$r_p$ ' + _MPC, ylabel=r'$\pi$ ' + _MPC,
        mirror=True, linthresh=1.0, shading='gouraud',
        levels=(0.2, 1, 10, 50, 100)),

    'xi_smu': StatSpec(
        source='xi_smu', ndim=2,
        xlabel=r'$s$ ' + _MPC, ylabel=r'$\mu$',
        shading='gouraud'),
}

def get_STATS():
    """Return a copy of the built-in statistics registry."""
    return dict(STATS)

def register_stat(name, **kwargs):
    """Add (or override) a statistic in the registry.

    >>> register_stat('xi_bar', source='xi_bar',
    ...               xlabel='$s$', ylabel=r'$\\bar\\xi$',
    ...               scale=lambda x, y: x**2 * y)
    """
    STATS[name] = StatSpec(**kwargs)
    return STATS[name]


# ----------------------------------------------------------------------
# Stat groups: one requested name expanding into several panels.
# A group is a callable ``(obj) -> list of stat names``, so the expansion
# can depend on the analysis object's configuration.
# ----------------------------------------------------------------------
def _xi_ells_group(obj):
    """Expand ``'xi_ells'`` into one panel per configured multipole.

    Reads the list of multipole orders from
    ``obj.args['clustering_settings']['xi_smu']['multipole_index']``,
    e.g. ``[0, 2, 4]``. Position ``i`` in that list is component ``i``
    of the array returned by ``get_xiells`` and carries order
    ``ells[i]``, so the panels are labelled :math:`\\xi_{\\ell}`
    accordingly.
    """
    ells = None
    try:
        ells = obj.args['clustering_settings']['xi_smu']['multipole_index']
    except (AttributeError, KeyError, TypeError):
        pass
    if ells is None:
        ells = [0, 2, 4]
    names = []
    for i, ell in enumerate(ells):
        ell = int(ell)
        name = f'xi{ell}'
        # component is the POSITION in the list, not the order itself
        register_stat(name, source='xi_ells', component=i,
                      xlabel=r'$s$ ' + _MPC,
                      ylabel=rf'$s \cdot \xi_{{{ell}}}(s)$ ' + _MPC,
                      scale=lambda x, y: x * y)
        names.append(name)
    return names


def _pk_ells_group(obj):
    """Expand ``'power_spectrum'`` into one panel per multipole.

    Uses the same configured multipole list as ``'xi_ells'``. If your
    ``compute_stats`` returns a single P(k) rather than a stack of
    multipoles, register a plain statistic instead::

        register_stat('power_spectrum', source='power_spectrum',
                      xlabel=r'$k$ [$h$/Mpc]', ylabel=r'$k P(k)$',
                      scale=lambda x, y: x * y)
    """
    ells = None
    try:
        ells = obj.args['clustering_settings']['xi_smu']['multipole_index']
    except (AttributeError, KeyError, TypeError):
        pass
    if ells is None:
        ells = [0, 2, 4]
    names = []
    for i, ell in enumerate(ells):
        ell = int(ell)
        name = f'pk{ell}'
        register_stat(name, source='power_spectrum', component=i,
                      xlabel=r'$k$ [$h/\mathrm{Mpc}$]',
                      ylabel=rf'$k \cdot P_{{{ell}}}(k)$ '
                             r'[$(\mathrm{Mpc}/h)^2$]',
                      scale=lambda x, y: x * y)
        names.append(name)
    return names


#: Names that expand into several statistics.
STAT_GROUPS = {
    'xi_ells': _xi_ells_group,
    'power_spectrum': _pk_ells_group,
}


def register_group(name, resolver):
    """Register a name that expands into several statistics.

    ``resolver`` is a callable ``(obj) -> list of stat names``.
    """
    STAT_GROUPS[name] = resolver
    return resolver


def _resolve_name(name, mapping):
    """Case-insensitive lookup: return the canonical key, or None."""
    if name in mapping:
        return name
    low = name.lower()
    for k in mapping:
        if k.lower() == low:
            return k
    return None


def _expand_stats(obj, stats):
    """Resolve names case-insensitively and expand any groups."""
    out = []
    for s in stats:
        grp = _resolve_name(s, STAT_GROUPS)
        if grp is not None:
            out.extend(STAT_GROUPS[grp](obj))
            continue
        key = _resolve_name(s, STATS)
        out.append(key if key is not None else s)
    return out


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _sigma(err):
    """Accept a 1D error vector or an (N, N) covariance; return sigma."""
    err = np.asarray(err)
    if err.ndim == 1:
        return err
    if err.ndim == 2 and err.shape[0] == err.shape[1]:
        return np.sqrt(np.diag(err))
    raise ValueError(f'error array of shape {err.shape} is neither (N,) nor (N, N)')


def _unpack_data(entry):
    """Normalise a user-supplied data entry to (x, y, sigma).

    Accepts a mapping with keys x/y/err (or cov), or a 2- or 3-tuple.
    """
    if entry is None:
        return None
    if isinstance(entry, Mapping):
        x, y = np.asarray(entry['x']), np.asarray(entry['y'])
        err = entry.get('err', entry.get('cov', entry.get('sigma')))
    else:
        entry = tuple(entry)
        if len(entry) == 3:
            x, y, err = entry
        elif len(entry) == 2:
            (x, y), err = entry, None
        else:
            raise ValueError('data entry must be (x, y) or (x, y, err)')
        x, y = np.asarray(x), np.asarray(y)
    return x, y, (None if err is None else _sigma(err))


def _mirror_quadrants(x, y, z):
    """Reflect a first-quadrant map about both axes.

    Parameters
    ----------
    x, y : 1D arrays
        Positive coordinate centres, e.g. ``r_p`` and ``pi``.
    z : 2D array, shape ``(len(x), len(y))``
        Values on the first quadrant.

    Returns
    -------
    x_full, y_full : 1D arrays of length ``2*len(x)``, ``2*len(y)``
    z_full : 2D array, shape ``(len(y_full), len(x_full))``
        Laid out for ``pcolormesh(x_full, y_full, z_full)``: rows follow
        ``y_full``, columns follow ``x_full``.
    """
    x, y, zt = np.asarray(x), np.asarray(y), np.asarray(z).T   # zt: (ny, nx)
    x_full = np.concatenate([-x[::-1], x])
    y_full = np.concatenate([-y[::-1], y])
    z_full = np.block([[zt[::-1, ::-1], zt[::-1, :]],
                       [zt[:, ::-1],    zt]])
    return x_full, y_full, z_full


def _unpack_result(entry, spec):
    """Turn one ``tt[source][tracer]`` entry into ``(x, y)`` or ``(x, y, z)``.

    Accepts the common shapes:
      * ``(x, y)``            -- 1D statistic
      * ``(x, y, z)``         -- 2D statistic (map)
      * a mapping with keys ``x``/``y`` (and ``z``)
      * a bare array         -- treated as ``y``, with ``x`` an index range
    """
    if isinstance(entry, Mapping):
        x = np.asarray(entry['x'])
        y = np.asarray(entry['y'])
        if spec.ndim == 2:
            return x, y, np.asarray(entry['z'])
        return x, y
    if isinstance(entry, (tuple, list)):
        if spec.ndim == 2:
            if len(entry) != 3:
                raise ValueError(
                    f"2D statistic '{spec.source}' expects (x, y, z), "
                    f'got a sequence of length {len(entry)}')
            return (np.asarray(entry[0]), np.asarray(entry[1]),
                    np.asarray(entry[2]))
        if len(entry) == 2:
            return np.asarray(entry[0]), np.asarray(entry[1])
        raise ValueError(
            f"1D statistic '{spec.source}' expects (x, y), got a sequence "
            f'of length {len(entry)}')
    arr = np.asarray(entry)                      # bare array: y only
    return np.arange(arr.shape[-1]), arr


def _fetch(obj, spec, cat, tracer_key, cache, tracers_arg):
    """Read one statistic for one tracer (or tracer pair) from compute_stats.

    Results are cached per source so ``compute_stats`` runs once per
    statistic rather than once per panel.
    """
    if spec.source not in cache:
        try:
            cache[spec.source] = obj.compute_stats(
                cat, stat=[spec.source], tracers=tracers_arg,
                **spec.getter_kwargs)
        except TypeError:       # compute_stats without a `tracers` argument
            cache[spec.source] = obj.compute_stats(
                cat, stat=[spec.source], **spec.getter_kwargs)
    tt = cache[spec.source]
    keys = [tracer_key] if isinstance(tracer_key, str) else list(tracer_key)

    # Peel the nesting one level at a time. `compute_stats` normally
    # returns tt[source][tracer], but it may return the per-tracer dict
    # directly (when a single statistic was requested) or even the bare
    # values (when there is a single tracer). Only descend while we are
    # still looking at a mapping, otherwise `k in block` would broadcast
    # over a numpy array and raise.
    block = tt
    if isinstance(block, Mapping) and spec.source in block:
        block = block[spec.source]

    if isinstance(block, Mapping):
        found = next((k for k in keys if k in block), None)
        if found is None:
            # Not every statistic is defined for every block -- e.g.
            # delta_sigma has no cross-tracer version. Signal "absent"
            # rather than failing, so the caller can skip that panel.
            return None
        entry = block[found]
    else:
        entry = block          # not nested by tracer: use it as-is

    if spec.prepare is not None:
        return spec.prepare(entry)
    out = _unpack_result(entry, spec)
    if spec.ndim == 2:
        return out
    x, y = out
    if spec.component is not None:
        y = np.asarray(y)[spec.component]
    return x, y

