"""Regression checks for Uchuu reader selection, state, and work partitioning."""

import pickle

import h5py
import numpy as np
import pytest

from HODDIES.sim_loader import Uchuu_io
from HODDIES.sim_loader.Uchuu_io import UchuuSim, _read_chunk_dde


@pytest.fixture(params=[False, True], ids=['original-case', 'lowercase'])
def rockstar_file(tmp_path, request):
    ids = 2**54 + np.arange(8, dtype=np.int64)
    source = dict(
        ID=ids, PID=np.array([-1, -1, -1, ids[0], -1, -1, -1, -1]),
        Mvir=np.array([1e13, 1e10, 1e12, 1e13, 1e14,
                       1e12 * (1 + 1e-8), 1e12 * (1 + 1e-6), 2e13]),
        Rvir=np.arange(8, dtype=float) + 200.,
        Rs=np.arange(8, dtype=float) + 20.,
        Vrms=np.arange(8, dtype=float) + 100.,
    )
    for index, column in enumerate(('X', 'Y', 'Z', 'VX', 'VY', 'VZ')):
        source[column] = np.arange(8, dtype=float) + .125 + index * 10
    directory = tmp_path / 'DDE' / 'DESIY1_DDE'
    directory.mkdir(parents=True)
    filename = directory / 'out_7.rockstar.h5'
    with h5py.File(filename, 'w') as handle:
        for name, values in source.items():
            handle[name.lower() if request.param else name] = values
    return tmp_path, filename, source


def configure(root, mass_cut=12.):
    simulation = UchuuSim.__new__(UchuuSim)
    simulation.init_logger(setup_logger=False)
    simulation.init_uchuu_args(path_to_sim=str(root), sim_name='DESIY1_DDE',
                               mass_cut=mass_cut, load_particles=True)
    simulation.z_simu = .49
    simulation.nthreads = 2
    return simulation


@pytest.mark.parametrize('mass_cut, rows', [(None, [0, 1, 2, 4, 5, 6, 7]),
                                          (12., [0, 4, 6, 7]), (20., [])])
def test_dde_preserves_selection_values_and_id_precision(rockstar_file, mass_cut, rows):
    _, filename, source = rockstar_file
    data, subs = _read_chunk_dde((0, 8), filename, mass_cut=mass_cut)
    assert subs is None
    assert data.dtype.names == ('x', 'y', 'z', 'vx', 'vy', 'vz', 'Rs', 'Rh',
                                'c', 'Mh', 'log10_Mh', 'Vrms', 'halo_id')
    assert data['halo_id'].dtype == np.dtype('i8')
    for name in data.dtype.names[:-1]:
        assert data[name].dtype == np.dtype('f4')
    np.testing.assert_array_equal(data['halo_id'], source['ID'][rows])
    for output, column in [('x', 'X'), ('y', 'Y'), ('z', 'Z'), ('vx', 'VX'),
                            ('vy', 'VY'), ('vz', 'VZ'), ('Rh', 'Rvir'),
                            ('Rs', 'Rs'), ('Mh', 'Mvir'), ('Vrms', 'Vrms')]:
        np.testing.assert_array_equal(data[output], source[column][rows].astype('f4'))
    np.testing.assert_array_equal(data['log10_Mh'], np.log10(source['Mvir'][rows].astype('f4')))
    np.testing.assert_array_equal(data['c'], source['Rvir'][rows].astype('f4') / source['Rs'][rows].astype('f4'))


def test_empty_selection_opens_once_and_skips_other_column_reads(rockstar_file, monkeypatch):
    _, filename, _ = rockstar_file
    opened, reads = [], []
    original_file = h5py.File
    original_getitem = h5py.Dataset.__getitem__

    def open_file(*args, **kwargs):
        opened.append(args[0])
        return original_file(*args, **kwargs)

    def getitem(dataset, key, *args, **kwargs):
        reads.append(dataset.name.lower())
        return original_getitem(dataset, key, *args, **kwargs)

    monkeypatch.setattr(h5py, 'File', open_file)
    monkeypatch.setattr(h5py.Dataset, '__getitem__', getitem)
    data, _ = _read_chunk_dde((0, 8), filename, mass_cut=20.)
    assert len(data) == 0
    assert len(opened) == 1
    assert reads == ['/mvir', '/pid']


def test_simulation_options_are_not_shared_between_instances(rockstar_file):
    root, filename, source = rockstar_file
    first = configure(root, mass_cut=12.)
    second = configure(root / 'other', mass_cut=13.)
    second.sim_name = 'Planck18'
    second.use_particles = False
    assert first.sim_name == 'DESIY1_DDE'
    assert first.use_particles is True
    assert first._root_dir_sim == str(root)
    data, _ = first.read_chunk_DDE((0, 8), filename)
    np.testing.assert_array_equal(data['halo_id'], source['ID'][[0, 4, 6, 7]])


class RecordingPool:
    """Execute real workers locally while checking submitted configuration."""
    instances = []

    def __init__(self, processes):
        self.processes = processes
        self.calls = []
        self.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def map(self, reader, chunks):
        chunks = list(chunks)
        self.calls.append((reader, chunks))
        # No bound simulation or shared class state is needed by a worker.
        reader = pickle.loads(pickle.dumps(reader))
        return [reader(chunk) for chunk in chunks]


@pytest.mark.parametrize('nchunks', [1, 3, 32])
def test_reader_limit_partition_and_dde_support(rockstar_file, monkeypatch, nchunks):
    root, _, source = rockstar_file
    monkeypatch.setattr(Uchuu_io.multiprocessing, 'Pool', RecordingPool)
    simulation = configure(root)
    simulation.read_Uchuu(nchuncks=nchunks)
    pool = RecordingPool.instances[-1]
    assert pool.processes == min(2, nchunks)
    reader, chunks = pool.calls[0]
    assert reader.keywords['mass_cut'] == 12.
    assert chunks[0][0] == 0 and chunks[-1][1] == 8
    assert all(a[1] == b[0] for a, b in zip(chunks[:-1], chunks[1:]))
    assert all(start < end for start, end in chunks)
    assert len(chunks) == min(8, nchunks)
    np.testing.assert_array_equal(simulation.hcat['halo_id'], source['ID'][[0, 4, 6, 7]])
    assert simulation.part_subsamples is None  # Original DDE capability is preserved.
    assert simulation.use_particles is False


def test_preprocessed_multifile_halos_and_subhalos(tmp_path, monkeypatch):
    directory = tmp_path / 'Uchuu_halo_catalogs' / 'halodir_040'
    directory.mkdir(parents=True)
    filenames = []
    for batch in range(2):
        filename = directory / f'part{batch}.h5'
        filenames.append(str(filename))
        with h5py.File(filename, 'w') as handle:
            for group in ('main', 'sub'):
                datasets = handle.create_group(group)
                datasets['halo_id'] = np.array([10, 20, 30], dtype='i8') + batch * 100
                datasets['log10_Mh'] = [11., 13., 14.]
                datasets['extra'] = np.array([4, 5, 6], dtype='i2')
    monkeypatch.setattr(Uchuu_io.multiprocessing, 'Pool', RecordingPool)
    monkeypatch.setattr(Uchuu_io.glob, 'glob', lambda pattern: filenames)
    simulation = configure(tmp_path)
    simulation.sim_name = 'Uchuu2Gpc'
    simulation.read_Uchuu(nchuncks=4)
    for catalogue in (simulation.hcat, simulation.part_subsamples):
        np.testing.assert_array_equal(catalogue['halo_id'], [20, 30, 120, 130])
        np.testing.assert_array_equal(catalogue['extra'], [5, 6, 5, 6])
        assert catalogue['extra'].dtype == np.dtype('i2')
    assert RecordingPool.instances[-1].processes == 2
    # The public chunk wrapper retains this instance's selection.
    np.testing.assert_array_equal(simulation.read_chuncks_2gpc((0, 3), filenames[0])['halo_id'], [20, 30])


def test_missing_file_and_invalid_chunk_count(tmp_path):
    simulation = configure(tmp_path)
    with pytest.raises(ValueError, match='nchuncks must be positive'):
        simulation.read_Uchuu(nchuncks=0)
    with pytest.raises(FileNotFoundError, match='No Uchuu catalogues match'):
        simulation.read_Uchuu()
