"""
Emulator verification plots: truth vs. prediction, one panel per
(statistic, tracer, component), laid out like `plot_stats`.

Panels are derived automatically from ``train_Dataset.slices``, so adding a
statistic to the training set adds panels here with no change to this code.
Blocks that hold several components (the multipoles of ``xi_ells`` or
``power_spectrum``, flattened into one slice) are split back into one panel
per multipole using the stored coordinate arrays.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import torch 


# ----------------------------------------------------------------------
# Per-statistic presentation
# ----------------------------------------------------------------------
_MPC = r'[$\mathrm{Mpc}/h$]'


@dataclass
class VerifSpec:
    """How one statistic is displayed."""
    xlabel: str = ''
    ylabel: str = ''                  # may contain '{ell}' for multipoles
    scale: Optional[Callable] = None  # (x, y) -> y_plot
    xscale: str = 'log'
    yscale: str = 'linear'
    ells: Optional[list] = None       # multipole orders, if a stacked block
    ndim: int = 1


VERIF = {
    'wp': VerifSpec(
        xlabel=r'$r_p$ ' + _MPC,
        ylabel=r'$r_p \cdot w_p(r_p)$',
        scale=lambda x, y: x * y),

    'xi_ells': VerifSpec(
        xlabel=r'$s$ ' + _MPC,
        ylabel=r'$s \cdot \xi_{{{ell}}}(s)$',
        scale=lambda x, y: x * y,
        ells=[0, 2, 4]),

    'power_spectrum': VerifSpec(
        xlabel=r'$k$ [$h/\mathrm{Mpc}$]',
        ylabel=r'$k \cdot P_{{{ell}}}(k)$',
        scale=lambda x, y: x * y,
        ells=[0, 2, 4]),

    'delta_sigma': VerifSpec(
        xlabel=r'$R$ ' + _MPC,
        ylabel=r'$R \cdot \Delta\Sigma(R)$',
        scale=lambda x, y: x * y),

    'CIC': VerifSpec(
        xlabel=r'$N_\mathrm{CIC}$',
        ylabel=r'$P(N_\mathrm{CIC})$',
        xscale='linear', yscale='log'),

    'xi_rppi': VerifSpec(xlabel=r'$r_p$ ' + _MPC,
                         ylabel=r'$\pi$ ' + _MPC, ndim=2),
    'xi_smu': VerifSpec(xlabel=r'$s$ ' + _MPC,
                        ylabel=r'$\mu$', ndim=2),
}


def register_verif(name, **kwargs):
    """Add or override how a statistic is displayed."""
    VERIF[name] = VerifSpec(**kwargs)
    return VERIF[name]


# ----------------------------------------------------------------------
# Panel discovery
# ----------------------------------------------------------------------
def _split_key(key, known_stats):
    """Split 'xi_ells_LRG' into ('xi_ells', 'LRG').

    Matched against the known statistic names rather than by splitting on
    '_', because tracer names may themselves contain underscores
    (e.g. 'ELG_LOPnotqso').
    """
    for stat in sorted(known_stats, key=len, reverse=True):
        if key.startswith(stat + '_'):
            return stat, key[len(stat) + 1:]
    return key, ''


def _stat_entry(ds, stat, tracer):
    """Return ``(coord_arrays, block_shape)`` for one (stat, tracer).

    Prefers ``ds.data_dict`` (the ``merged`` dict returned by
    ``load_data``), whose entries are ``[coord1, ..., coordN, values]``
    with ``values`` of shape ``(n_samples, *block_shape)``. Falls back to
    the separate ``ds.coords`` / ``ds.block_shapes`` attributes.
    """
    dd = getattr(ds, 'data_dict', None)
    if dd is not None and stat in dd and tracer in dd[stat]:
        entry = dd[stat][tracer]
        coords = [np.asarray(a) for a in entry[:-1]]
        shape = np.asarray(entry[-1]).shape[1:]
        return coords, shape

    coords = getattr(ds, 'coords', {}).get(stat, {}).get(tracer)
    shape = getattr(ds, 'block_shapes', {}).get(f'{stat}_{tracer}')
    return coords, shape


def _panels(ds, stats=None):
    """Build the panel list from ``ds.slices``.

    Returns a list of dicts with the tracer, a display label, the slice
    into the data vector, the x coordinates, and the presentation spec.
    """
    known = list(getattr(ds, 'stats', []))
    out = []

    for key, sl in ds.slices.items():
        stat, tracer = _split_key(key, known)
        if stats is not None and stat not in stats:
            continue
        spec = VERIF.get(stat, VerifSpec(xlabel='index', ylabel=stat))

        xs, shape = _stat_entry(ds, stat, tracer)
        width = sl.stop - sl.start

        if spec.ndim == 2:
            out.append(dict(tracer=tracer, stat=stat, label=stat,
                            sl=sl, x=xs, spec=spec, comp=None, shape=shape))
            continue

        x = np.asarray(xs[0]) if xs else np.arange(width)
        ncomp = max(1, width // len(x))

        if ncomp == 1:
            out.append(dict(tracer=tracer, stat=stat, label=stat,
                            sl=sl, x=x, spec=spec, comp=None))
        else:
            ells = (spec.ells or list(range(ncomp)))[:ncomp]
            for c in range(ncomp):
                sub = slice(sl.start + c * len(x), sl.start + (c + 1) * len(x))
                out.append(dict(
                    tracer=tracer, stat=stat, comp=c, ell=ells[c],
                    label=f'{stat}_{ells[c]}', sl=sub, x=x, spec=spec))
    return out


def _ylabel(p):
    lab = p['spec'].ylabel
    if '{ell}' in lab:
        return lab.format(ell=p.get('ell', p.get('comp', 0)))
    return lab


# ----------------------------------------------------------------------
# Emulator predictions on the test set
# ----------------------------------------------------------------------
def _predict_test_set(train_Dataset, model, show_normalized=False, rescale_var=1):
    """Run ``model`` on ``train_Dataset``'s test set and denormalize.

    Returns ``(X_test, Y_test, Y_pred, err_pred)`` where ``Y_test``/``Y_pred``
    are in physical units unless ``show_normalized`` is set, in which case
    they are left in the normalizer's internal scale.
    """
    import torch
    X_test = train_Dataset.X_test
    Y_test = train_Dataset.y_test if not show_normalized else train_Dataset.y_test_norm
    x_norm = torch.tensor(train_Dataset.x_test_norm, dtype=torch.float32)

    Y_pred, var_pred = model.predict(x_norm, no_grad=True)
    if not show_normalized:
        Y_pred, var_pred = train_Dataset.normalizer.denormalize_y(
            Y_pred.cpu().numpy(), var_pred.cpu().numpy())
    else:
        Y_pred = Y_pred.cpu().numpy()
        var_pred = var_pred.cpu().numpy()
    err_pred = np.sqrt(np.asarray(var_pred)) * rescale_var
    return X_test, np.asarray(Y_test), np.asarray(Y_pred), err_pred


def _block_name(p):
    """Short display name for a panel, e.g. 'xi_0_LRG' instead of 'xi_ells_0_LRG'."""
    stat = p['stat'][:-5] if p['stat'].endswith('_ells') else p['stat']
    if p.get('comp') is not None:
        stat = f"{stat}_{p['ell']}"
    return f'{stat}_{p["tracer"]}' if p['tracer'] else stat


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def plot_verif(train_Dataset, model, nb_plots=5, stats=None,
               std_emu=None, rescale_var=1, show_normalized=False,
               max_cols=4, indices=None, seed=None,
               fontsize=11, residual_ylim=(-5, 5), block_hspace=0.55,
               wspace=0.30, height_ratios=(3, 1), colors=None, show=True):
    """Compare emulator predictions with the test set.

    One figure per test sample, one row of panels per tracer, wrapped at
    ``max_cols`` columns, with a ``(truth - prediction)/sigma`` sub-panel
    flush beneath each panel.

    Parameters
    ----------
    train_Dataset : Training_DatasetManager
        Must expose ``slices``; ``coords`` (see note in the module
        docstring) is needed to split multipole blocks into panels.
    model : object with ``predict``
    nb_plots : int
        Number of random test samples to show, ignored when ``indices``
        is given.
    stats : sequence of str or None
        Restrict to these statistics. ``None`` plots everything in
        ``slices``.
    indices : sequence of int or None
        Explicit test-set indices to plot.
    """
    # ---- predictions -------------------------------------------------
    X_test, Y_test, Y_pred, err_pred = _predict_test_set(
        train_Dataset, model, show_normalized=show_normalized, rescale_var=rescale_var)

    # ---- panels ------------------------------------------------------
    panels = _panels(train_Dataset, stats)
    if not panels:
        raise ValueError(
            f'no panels for stats={stats}; available slices: '
            f'{sorted(train_Dataset.slices)}')

    tracers = list(dict.fromkeys(p['tracer'] for p in panels))
    per_tracer = {t: [p for p in panels if p['tracer'] == t] for t in tracers}
    npanel = max(len(v) for v in per_tracer.values())
    ncol = max(1, min(max_cols, npanel))
    nsub = int(np.ceil(npanel / ncol))
    nblock = len(tracers)

    default_colors = {'ELG': 'deepskyblue', 'QSO': 'seagreen',
                      'LRG': 'red', 'BGS': 'goldenrod'}
    colors = {**default_colors, **(colors or {})}

    # ---- which samples ----------------------------------------------
    rng = np.random.default_rng(seed)
    if indices is None:
        n_avail = Y_pred.shape[0]
        indices = rng.choice(np.arange(n_avail),
                             min(nb_plots, n_avail), replace=False)

    figs = []
    for i in indices:
        fig = plt.figure(figsize=(4.6 * ncol,
                                  4.4 * nsub * nblock + 0.9 * (nsub * nblock - 1)))
        outer = GridSpec(nsub * nblock, ncol, figure=fig,
                         hspace=block_hspace, wspace=wspace,
                         left=0.08, right=0.98, top=0.92, bottom=0.08)

        for b, tracer in enumerate(tracers):
            plist = per_tracer[tracer]
            color = colors.get(tracer, f'C{b}')

            for j, p in enumerate(plist):
                r, c = b * nsub + j // ncol, j % ncol
                sl, x, spec = p['sl'], p['x'], p['spec']

                if spec.ndim == 2:      # maps get the full cell
                    ax = fig.add_subplot(outer[r, c])
                    _plot_2d_residual(fig, ax, None, p, Y_test[i], Y_pred[i],
                                      err_pred[i][sl], fontsize)
                    if j == 0 and ax.get_visible():
                        ax.set_title(tracer, fontsize=fontsize + 1, loc='left')
                    continue

                inner = outer[r, c].subgridspec(
                    2, 1, height_ratios=height_ratios, hspace=0.0)
                ax = fig.add_subplot(inner[0])
                rax = fig.add_subplot(inner[1], sharex=ax)
                ax.tick_params(labelbottom=False)

                yt = Y_test[i][sl]
                yp = Y_pred[i][sl]
                ep = err_pred[i][sl]

                f = (lambda v: spec.scale(x, v)) if spec.scale else (lambda v: v)

                ax.plot(x, f(yt), lw=2, color=color, label='Truth')
                ax.plot(x, f(yp), ls='--', lw=1.6, color='k',
                        label='Prediction')
                ax.fill_between(x, f(yp - ep), f(yp + ep), alpha=0.3,
                                color='green', label=r'$\sigma_\mathrm{emu}$')
                if std_emu is not None:
                    se = std_emu[sl]
                    ax.fill_between(x, f(yp - se), f(yp + se), alpha=0.3,
                                    color='red', label='pred. error')

                ax.set_ylabel(_ylabel(p), fontsize=fontsize)
                ax.set_xscale(spec.xscale)
                ax.set_yscale(spec.yscale)
                ax.grid(alpha=0.25)
                if j == 0:
                    ax.set_title(tracer, fontsize=fontsize + 1, loc='left')
                if j == len(plist) - 1:
                    ax.legend(fontsize=fontsize - 1)

                with np.errstate(divide='ignore', invalid='ignore'):
                    res = np.where(ep != 0, (yt - yp) / ep, np.nan)
                rax.plot(x, res, color='k', lw=1.3)
                rax.axhline(0, ls='--', color='grey', lw=1)
                rax.axhspan(-1, 1, color='grey', alpha=0.2)
                rax.set_ylim(*residual_ylim)
                rax.set_xscale(spec.xscale)
                rax.set_xlabel(spec.xlabel, fontsize=fontsize)
                rax.set_ylabel(r'$\delta/\sigma$', fontsize=fontsize)
                rax.grid(alpha=0.25)

        # parameter values of this sample as the title
        try:
            names = train_Dataset.name_arr
            vals = X_test[i]
            txt = ', '.join(f'{n} = {v:.2f}' for n, v in zip(names, vals))
            fig.suptitle(txt, fontsize=fontsize + 1)
        except Exception:
            pass

        figs.append(fig)
        if show:
            fig.tight_layout()
            plt.show()
    return figs


def plot_error_diagnostics(train_Dataset, model, stats=None, show_normalized=False,
                           rescale_var=1, target_frac_err=0.01, residual_ylim=None,
                           figsize=(9, 4), show=True):
    """Emulator error diagnostics over the full test set.

    Three figures summarizing prediction error across the whole test set
    (as opposed to `plot_verif`, which shows individual test samples):

    1. per-bin fractional error, median with 68%/95% bands
    2. predicted vs. true scatter, one panel per statistic block
    3. per-bin residual bias, mean +/- std

    Parameters
    ----------
    train_Dataset : Training_DatasetManager
        Must expose ``slices``/``stats`` (see `plot_verif`).
    model : object with ``predict``
    stats : sequence of str or None
        Restrict to these statistics. ``None`` uses everything in
        ``slices``.
    target_frac_err : float or None
        Reference line drawn on the fractional-error plot. ``None`` omits it.
    residual_ylim : (float, float) or None
        y-limits for the residual-bias plot. ``None`` autoscales.

    Returns
    -------
    (fig_frac_err, fig_pred_vs_true, fig_residual_bias)
    """
    _, Y_test, Y_pred, _ = _predict_test_set(
        train_Dataset, model, show_normalized=show_normalized, rescale_var=rescale_var)

    panels = [p for p in _panels(train_Dataset, stats) if p['spec'].ndim != 2]
    if not panels:
        raise ValueError(
            f'no 1D panels for stats={stats}; available slices: '
            f'{sorted(train_Dataset.slices)}')
    panels = sorted(panels, key=lambda p: p['sl'].start)
    boundaries = [p['sl'].start for p in panels] + [panels[-1]['sl'].stop]

    resid = Y_pred - Y_test
    with np.errstate(divide='ignore', invalid='ignore'):
        frac = np.abs(resid) / np.abs(Y_test)
    x = np.arange(Y_pred.shape[1])

    # ---------- 1. Per-bin fractional error (median + 68/95 band) -----
    med = np.nanmedian(frac, axis=0)
    lo68, hi68 = np.nanpercentile(frac, [16, 84], axis=0)
    hi95 = np.nanpercentile(frac, 97.5, axis=0)

    fig1, ax = plt.subplots(figsize=figsize)
    ax.fill_between(x, lo68, hi68, alpha=0.3, label='68%')
    ax.plot(x, med, lw=1.5, label='median')
    ax.plot(x, hi95, lw=1, ls='--', label='95%')
    for b in boundaries[1:-1]:
        ax.axvline(b, color='k', ls=':', alpha=0.5)
    if target_frac_err is not None:
        ax.axhline(target_frac_err, color='r', ls='--', alpha=0.5)
    ax.set_yscale('log')
    ax.set_xlabel('bin')
    ax.set_ylabel('fractional error')
    ax.set_title('Per-bin fractional error')
    ax.legend()
    fig1.tight_layout()

    # ---------- 2. Predicted vs true (one panel per statistic) --------
    n = len(panels)
    fig2, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    axes = np.atleast_1d(axes)
    for ax, p in zip(axes, panels):
        t, y = Y_test[:, p['sl']].ravel(), Y_pred[:, p['sl']].ravel()
        ax.scatter(t, y, s=3, alpha=0.3)
        lim = [min(t.min(), y.min()), max(t.max(), y.max())]
        ax.plot(lim, lim, 'k--', lw=1)
        ax.set_xlabel('true')
        ax.set_ylabel('pred')
        ax.set_title(_block_name(p))
    fig2.tight_layout()

    # ---------- 3. Residual (bias) per bin: mean +/- std ---------------
    fig3, ax = plt.subplots(figsize=figsize)
    mean_r, std_r = resid.mean(axis=0), resid.std(axis=0)

    ax.plot(x, mean_r, lw=1.5, label='mean residual (bias)')
    ax.fill_between(x, mean_r - std_r, mean_r + std_r, alpha=0.3, label=r'$\pm 1\sigma$')
    ax.axhline(0, color='k', lw=0.8)
    for b in boundaries[1:-1]:
        ax.axvline(b, color='k', ls=':', alpha=0.5)
    if residual_ylim is not None:
        ax.set_ylim(*residual_ylim)
    ax.set_xlabel('bin')
    ax.set_ylabel('pred - true')
    ax.set_title('Residual bias per bin')
    ax.legend()
    fig3.tight_layout()

    if show:
        plt.show()
    return fig1, fig2, fig3


def _plot_2d_residual(fig, ax, rax, p, yt_full, yp_full, err_pred, fontsize):
    """Residual map for a 2D block.

    Uses ``(truth - prediction)/sigma_emu`` where the emulator error is
    available and non-zero, otherwise falls back to the relative
    difference ``(truth - prediction)/truth``.
    """
    from matplotlib.colors import TwoSlopeNorm
    if rax is not None:
        rax.set_visible(False)
    sl, spec = p['sl'], p['spec']
    shape = p.get('shape')
    xs = p['x']
    if shape is None or xs is None or len(xs) < 2:
        ax.set_visible(False)
        return

    yt, yp = np.asarray(yt_full[sl], float), np.asarray(yp_full[sl], float)
    ep = np.asarray(err_pred[sl], float) if err_pred is not None else None

    use_sigma = ep is not None and np.any(np.isfinite(ep) & (ep != 0))
    with np.errstate(divide='ignore', invalid='ignore'):
        if use_sigma:
            res = np.where(np.isfinite(ep) & (ep != 0), (yt - yp) / ep, np.nan)
            cbar_label = r'$(\mathrm{true}-\mathrm{pred})/\sigma_\mathrm{emu}$'
            vlim = 3.0
        else:
            res = np.where(yt != 0, (yt - yp) / yt, np.nan)
            cbar_label = r'$(\mathrm{true}-\mathrm{pred})/\mathrm{true}$'
            finite = res[np.isfinite(res)]
            # symmetric scale from the data, capped so outliers don't
            # flatten the whole map
            vlim = float(np.nanpercentile(np.abs(finite), 99)) if finite.size else 1.0
            vlim = max(vlim, 1e-6)
    
    res = res.reshape(shape)
    x, y = np.asarray(xs[0], float), np.asarray(xs[1], float)
    gx, gy = np.isfinite(x), np.isfinite(y)
    if not (gx.all() and gy.all()):
        res, x, y = res[np.ix_(gx, gy)], x[gx], y[gy]
    if x.size < 2 or y.size < 2 or not np.isfinite(res).any():
        ax.set_visible(False)
        return

    m = ax.pcolormesh(x, y, np.ma.masked_invalid(res.T), cmap='RdBu_r',
                      norm=TwoSlopeNorm(0.0, -vlim, vlim), shading='auto')
    fig.colorbar(m, ax=ax, label=cbar_label)
    ax.set_xscale(spec.xscale)
    ax.set_xlabel(spec.xlabel, fontsize=fontsize)
    ax.set_ylabel(spec.ylabel, fontsize=fontsize)


def plot_bf_results(train_Dataset, model, x_best_fit, data_vec, err_data_vec,x_truth=None, stats=None,
               show_normalized=False, save_fn=None,
               max_cols=4,
               fontsize=11, residual_ylim=(-5, 5), block_hspace=0.55,
               wspace=0.30, height_ratios=(3, 1), colors=None, show=True):
    """Compare emulator predictions with the test set.

    One figure per test sample, one row of panels per tracer, wrapped at
    ``max_cols`` columns, with a ``(truth - prediction)/sigma`` sub-panel
    flush beneath each panel.

    Parameters
    ----------
    train_Dataset : Training_DatasetManager
        Must expose ``slices``; ``coords`` (see note in the module
        docstring) is needed to split multipole blocks into panels.
    model : object with ``predict``
    stats : sequence of str or None
        Restrict to these statistics. ``None`` plots everything in
        ``slices``.
    indices : sequence of int or None
        Explicit test-set indices to plot.
    """
    # ---- predictions -------------------------------------------------
    if x_truth is not None:
        x_norm_truth = torch.tensor(train_Dataset.normalizer.normalize_x(x_truth), dtype=torch.float32)
        Y_pred_truth, var_pred_truth = model.predict(x_norm_truth, no_grad=True)
        if not show_normalized:
            Y_pred_truth, var_pred_truth = train_Dataset.normalizer.denormalize_y(
                Y_pred_truth.cpu().numpy(), var_pred_truth.cpu().numpy())
        else:
            Y_pred_truth = np.asarray(Y_pred_truth.cpu().numpy())
            var_pred_truth = np.asarray(var_pred_truth.cpu().numpy())
        err_pred_truth = np.sqrt(np.asarray(var_pred_truth))

    x_norm_bf = torch.tensor(train_Dataset.normalizer.normalize_x(x_best_fit), dtype=torch.float32)
    Y_pred_bf, var_pred_bf = model.predict(x_norm_bf, no_grad=True)
    if not show_normalized:
        Y_pred_bf, var_pred_bf = train_Dataset.normalizer.denormalize_y(
            Y_pred_bf.cpu().numpy(), var_pred_bf.cpu().numpy())
    else:
        Y_pred_bf = np.asarray(Y_pred_bf.cpu().numpy())
        var_pred_bf = np.asarray(var_pred_bf.cpu().numpy())
    err_pred_bf = np.sqrt(np.asarray(var_pred_bf))
        
    # ---- panels ------------------------------------------------------
    from HODDIES.fit_functions.plotting_emulator_func import _plot_2d_residual, _ylabel, _panels
    panels = _panels(train_Dataset, stats)
    if not panels:
        raise ValueError(
            f'no panels for stats={stats}; available slices: '
            f'{sorted(train_Dataset.slices)}')

    tracers = list(dict.fromkeys(p['tracer'] for p in panels))
    per_tracer = {t: [p for p in panels if p['tracer'] == t] for t in tracers}
    npanel = max(len(v) for v in per_tracer.values())
    ncol = max(1, min(max_cols, npanel))
    nsub = int(np.ceil(npanel / ncol))
    nblock = len(tracers)

    default_colors = {'ELG': 'deepskyblue', 'QSO': 'seagreen',
                      'LRG': 'red', 'BGS': 'goldenrod'}
    colors = {**default_colors, **(colors or {})}

    # ---- which samples ----------------------------------------------

    
    fig = plt.figure(figsize=(4.6 * ncol,
                                4.4 * nsub * nblock + 0.9 * (nsub * nblock - 1)))
    outer = GridSpec(nsub * nblock, ncol, figure=fig,
                        hspace=block_hspace, wspace=wspace,
                        left=0.08, right=0.98, top=0.92, bottom=0.08)

    for b, tracer in enumerate(tracers):
        plist = per_tracer[tracer]
        color = colors.get(tracer, f'C{b}')

        for j, p in enumerate(plist):
            r, c = b * nsub + j // ncol, j % ncol
            sl, x, spec = p['sl'], p['x'], p['spec']

            if spec.ndim == 2:      # maps get the full cell
                ax = fig.add_subplot(outer[r, c])
                _plot_2d_residual(fig, ax, None, p, data_vec, Y_pred_bf[sl],
                                    err_pred_bf[sl], fontsize)
                if j == 0 and ax.get_visible():
                    ax.set_title(tracer, fontsize=fontsize + 1, loc='left')
                continue

            inner = outer[r, c].subgridspec(
                2, 1, height_ratios=height_ratios, hspace=0.0)
            ax = fig.add_subplot(inner[0])
            rax = fig.add_subplot(inner[1], sharex=ax)
            ax.tick_params(labelbottom=False)



            f = (lambda v: spec.scale(x, v)) if spec.scale else (lambda v: v)

            if x_truth is not None:
                ax.plot(x, f(Y_pred_truth[sl]), lw=2, color='firebrick', label='Truth')
                ax.fill_between(x, f(Y_pred_truth[sl] - err_pred_truth[sl]), f(Y_pred_truth[sl] + err_pred_truth[sl]), alpha=0.3,
                                            color='firebrick', label=r'$\sigma_\mathrm{emu}$')
            
            ax.plot(x, f(Y_pred_bf[sl]), ls='--', lw=1.6, color=color, label='Best fit')
            ax.fill_between(x, f(Y_pred_bf[sl] - err_pred_bf[sl]), f(Y_pred_bf[sl] + err_pred_bf[sl]), alpha=0.3,
                            color=color, label=r'$\sigma_\mathrm{emu}$')

            ax.errorbar(x, f(data_vec[sl]), f(err_data_vec[sl]), fmt='o', color=color, label='Data')

            
            ax.set_ylabel(_ylabel(p), fontsize=fontsize)
            ax.set_xscale(spec.xscale)
            ax.set_yscale(spec.yscale)
            ax.grid(alpha=0.25)
            if j == 0:
                ax.set_title(tracer, fontsize=fontsize + 1, loc='left')
            if j == len(plist) - 1:
                ax.legend(fontsize=fontsize - 1)

            with np.errstate(divide='ignore', invalid='ignore'):
                err_bf = np.sqrt(err_pred_bf[sl]**2 + err_data_vec[sl]**2)
                res = np.where(err_bf != 0, (data_vec[sl] - Y_pred_bf[sl]) / err_bf, np.nan)
            rax.plot(x, res, color=color, lw=1.3)
            if x_truth is not None:
                err_truth = np.sqrt(err_pred_truth[sl]**2 + err_data_vec[sl]**2)
                res_truth = np.where(err_truth != 0, (data_vec[sl] - Y_pred_truth[sl]) / err_truth, np.nan)
                rax.plot(x, res_truth, color='firebrick', lw=1.3, ls='--')
            
            rax.axhline(0, ls='--', color='grey', lw=1)
            rax.axhspan(-1, 1, color='grey', alpha=0.2)
            rax.set_ylim(*residual_ylim)
            rax.set_xscale(spec.xscale)
            rax.set_xlabel(spec.xlabel, fontsize=fontsize)
            rax.set_ylabel(r'$\delta/\sigma$', fontsize=fontsize)
            rax.grid(alpha=0.25)

        if show:
            fig.tight_layout()
            plt.show()
        if save_fn:
            fig.tight_layout()
            fig.savefig(save_fn, dpi=600, bbox_inches='tight')
    return fig