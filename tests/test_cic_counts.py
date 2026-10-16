"""Regression checks for independent counts-in-cylinder tracer histograms."""

import unittest
from unittest.mock import patch

import numpy as np

from HODDIES import hod as hod_module


class TestCICCounts(unittest.TestCase):
    def setUp(self):
        self.hod = hod_module.HOD.__new__(hod_module.HOD)
        self.hod.boxsize = 100.
        self.hod.args = {'clustering_settings': {
            'rsd': False, 'CIC': {'R_max': 5., 'L_max': 15.}}}
        # Encode deterministic neighbour counts in x; only the estimator is
        # replaced, leaving catalogue selection and histogramming unchanged.
        self.cat = np.array([
            ('LRG', 0., 0., 0.), ('LRG', 1., 0., 0.), ('LRG', 1., 0., 0.),
            ('ELG', 2., 0., 0.), ('ELG', 4., 0., 0.),
        ], dtype=[('TRACER', 'U3'), ('x', 'f8'), ('y', 'f8'), ('z', 'f8')])
        estimator = patch.object(hod_module, 'compute_CIC',
                                 side_effect=lambda x, *args: x.astype(int))
        self.compute = estimator.start()
        self.addCleanup(estimator.stop)

    def test_unequal_tracer_sizes_and_later_larger_counts(self):
        result = self.hod.get_CIC(self.cat, tracers=['LRG', 'ELG'],
                                  return_dic=True, density=False)
        for tracer in ['LRG', 'ELG']:
            np.testing.assert_array_equal(result[tracer + '_' + tracer][0],
                                          np.arange(5))
        np.testing.assert_array_equal(result['LRG_LRG'][1], [1, 2, 0, 0, 0])
        np.testing.assert_array_equal(result['ELG_ELG'][1], [0, 0, 1, 0, 1])
        self.assertEqual(self.compute.call_count, 2)

    def test_equal_tracer_sizes_do_not_mix_distributions(self):
        cat = self.cat[[0, 1, 3, 4]]
        result = self.hod.get_CIC(cat, tracers=['LRG', 'ELG'],
                                  return_dic=True, density=False)
        np.testing.assert_array_equal(result['LRG_LRG'][1], [1, 1, 0, 0, 0])
        np.testing.assert_array_equal(result['ELG_ELG'][1], [0, 0, 1, 0, 1])

    def test_each_tracer_is_normalized_independently(self):
        result = self.hod.get_CIC(self.cat, tracers=['LRG', 'ELG'], return_dic=True)
        np.testing.assert_allclose(result['LRG_LRG'][1], [1/3, 2/3, 0, 0, 0])
        np.testing.assert_allclose(result['ELG_ELG'][1], [0, 0, .5, 0, .5])

    def test_configured_and_explicit_count_limits(self):
        self.hod.args['clustering_settings']['CIC']['max_count'] = 2
        for limit, expected in [(None, [0, 0, 1]), (4, [0, 0, 1, 0, 1])]:
            result = self.hod.get_CIC(self.cat, tracers=['LRG', 'ELG'],
                                      return_dic=True, density=False, max_count=limit)
            np.testing.assert_array_equal(result['ELG_ELG'][1], expected)

    def test_single_tracer_return_and_raw_counts(self):
        centers, hist, counts = self.hod.get_CIC(
            self.cat, tracers='LRG', return_counts=True, density=False)
        np.testing.assert_array_equal(centers, [0, 1])
        np.testing.assert_array_equal(hist, [1, 2])
        self.assertEqual(len(counts), 1)
        np.testing.assert_array_equal(counts[0], [0, 1, 1])

    def test_multiple_histograms_and_raw_counts_follow_requested_order(self):
        centers, hists, counts = self.hod.get_CIC(
            self.cat, tracers=['ELG', 'LRG'], return_counts=True, density=False)
        np.testing.assert_array_equal(centers, np.arange(5))
        np.testing.assert_array_equal(hists[0], [0, 0, 1, 0, 1])
        np.testing.assert_array_equal(hists[1], [1, 2, 0, 0, 0])
        np.testing.assert_array_equal(counts[0], [2, 4])
        np.testing.assert_array_equal(counts[1], [0, 1, 1])


if __name__ == '__main__':
    unittest.main()
