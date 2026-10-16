"""Fitting vectors preserve pair/statistic/component order across result shapes."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from HODDIES.fit_functions.fits_functions import flatten_stats, func_stochopy


class TestFlattenStats(unittest.TestCase):
    def test_explicit_pair_and_statistic_order_with_auto_only_lensing(self):
        # Deliberately insert dictionaries in a different order from the fit.
        result = {
            'delta_sigma': {'B_B': ([999], [90]), 'A_A': ([999], [30])},
            'xi_ells': {
                'B_B': ([999], [[70], [80]]),
                'A_B': ([999], [[50], [60]]),
                'A_A': ([999], [[10], [20]]),
            },
            'wp': {'B_B': ([999], [65]), 'A_B': ([999], [40]),
                   'A_A': ([999], [5])},
        }
        vector = flatten_stats(result, ['wp', 'xi_ells', 'delta_sigma'],
                               ['A_A', 'A_B', 'B_B'])
        np.testing.assert_array_equal(vector, [5, 10, 20, 30, 40, 50, 60, 65, 70, 80, 90])
        self.assertEqual(vector.ndim, 1)

    def test_infers_statistics_and_pairs_in_dictionary_order(self):
        result = {
            'xi_ells': {'B_B': (None, [[1, 2], [3, 4]]), 'A_A': (None, [[5], [6]])},
            'wp': {'A_A': (None, [7]), 'B_B': (None, [8])},
        }
        np.testing.assert_array_equal(flatten_stats(result), [1, 2, 3, 4, 8, 5, 6, 7])

    def test_inferred_pairs_include_pairs_missing_from_first_statistic(self):
        result = {
            'delta_sigma': {'A_A': (None, [1]), 'B_B': (None, [2])},
            'wp': {'A_A': (None, [3]), 'A_B': (None, [4]), 'B_B': (None, [5])},
        }
        # First-seen pair order is A_A, B_B, A_B; every pair is included.
        np.testing.assert_array_equal(flatten_stats(result), [1, 3, 2, 5, 4])

    def test_infers_pairs_only_from_selected_statistics(self):
        result = {'wp': {'A_A': (None, [1])},
                  'xi0': {'B_B': (None, [2]), 'A_A': (None, [3])}}
        np.testing.assert_array_equal(flatten_stats(result, stats='xi0'), [2, 3])

    def test_explicit_pair_order_with_inferred_statistics(self):
        result = {'wp': {'B_B': (None, [1]), 'A_A': (None, [2])},
                  'xi0': {'B_B': (None, [3]), 'A_A': (None, [4])}}
        np.testing.assert_array_equal(flatten_stats(result, comb_trs=['A_A', 'B_B']),
                                      [2, 4, 1, 3])

    def test_inference_with_empty_results_or_statistic_blocks(self):
        for result in [{}, {'wp': {}}]:
            self.assertEqual(flatten_stats(result).shape, (0,))

    def test_any_number_of_multipoles_and_dimensions(self):
        for shape in [(), (5,), (1, 5), (4, 5), (2, 3, 5)]:
            values = np.arange(int(np.prod(shape))).reshape(shape)
            vector = flatten_stats({'stat': {'A_A': (None, values)}}, 'stat', 'A_A')
            np.testing.assert_array_equal(vector, values.ravel())

    def test_component_order_is_preserved(self):
        # Stored order could be ell=4, ell=0, ell=2; flattening must not sort.
        result = {'xi_ells': {'A_A': ([1, 2], [[40, 41], [0, 1], [20, 21]])}}
        np.testing.assert_array_equal(flatten_stats(result, 'xi_ells', 'A_A'),
                                      [40, 41, 0, 1, 20, 21])

    def test_two_dimensional_maps_exclude_both_coordinate_arrays(self):
        result = {'xi_smu': {'A_A': ([100, 200], [300, 400], [[1, 2], [3, 4]])}}
        np.testing.assert_array_equal(flatten_stats(result, 'xi_smu', 'A_A'), [1, 2, 3, 4])

    def test_ragged_components_and_object_arrays(self):
        values = [np.array([1., 2.]), np.array([3.]), np.array([4., 5., 6.])]
        objects = np.empty(3, dtype=object)
        objects[:] = values
        for components in [values, objects]:
            result = {'xi_ells': {'A_A': (None, components)}}
            np.testing.assert_array_equal(flatten_stats(result, 'xi_ells', 'A_A'),
                                          [1, 2, 3, 4, 5, 6])

    def test_bare_arrays_are_values_not_coordinate_sequences(self):
        result = {'CIC': {'A_A': np.array([2, 0, 3])}}
        np.testing.assert_array_equal(flatten_stats(result, 'CIC', 'A_A'), [2, 0, 3])

    def test_pair_sequences_and_names_with_underscores(self):
        pairs = np.array([['ELG_LOP', 'LRG'], ['LRG', 'LRG']])
        result = {'wp': {'LRG_LRG': (None, [2]), 'ELG_LOP_LRG': (None, [1])}}
        np.testing.assert_array_equal(flatten_stats(result, ['wp'], pairs), [1, 2])

    def test_missing_statistic_raises(self):
        with self.assertRaises(KeyError):
            flatten_stats({'wp': {}}, ['wp', 'xi_ells'], ['A_A'])

    def test_empty_selections_and_blocks(self):
        for result, stats, pairs in [({}, [], ['A_A']), ({'wp': {}}, ['wp'], []),
                                     ({'wp': {}}, ['wp'], ['A_A']),
                                     ({'wp': {'A_A': (None, [])}}, ['wp'], ['A_A'])]:
            vector = flatten_stats(result, stats, pairs)
            self.assertEqual(vector.shape, (0,))

    def test_matches_previous_vector_for_complete_rectangular_results(self):
        stats, pairs = ['wp', 'xi_ells'], ['B_B', 'A_B', 'A_A']
        result = {stat: {pair: (None, np.arange((j + 1) * 3).reshape(j + 1, 3) + i * 10)
                         for i, pair in enumerate(pairs)} for j, stat in enumerate(stats)}
        previous = np.hstack([np.hstack([np.hstack(result[stat][pair][-1])
                                        for stat in stats]) for pair in pairs])
        np.testing.assert_array_equal(flatten_stats(result, stats, pairs), previous)

    def test_minimizer_uses_flattened_vector_for_chi_squared(self):
        result = {'wp': {'A_B': (None, [3]), 'A_A': (None, [1, 2])},
                  'delta_sigma': {'A_A': (None, [4])}}
        hod = SimpleNamespace(
            name_params=['alpha_A'], priors=np.array([[0., 1.]]),
            logger=Mock(), update_new_param=Mock(), make_mock_cat=Mock(return_value='cat'),
            compute_stats=Mock(return_value=result), _stats_to_fit=['wp', 'delta_sigma'],
            _comb_tr_list=['A_A', 'A_B'], _tracer_to_fit=['A', 'B'],
            data=np.zeros(4), inv_cov2=np.diag([1., 2., 3., 4.]),
        )
        # Required vector is [1, 2, 4, 3]; the cross has no delta_sigma.
        self.assertEqual(func_stochopy([.5], hod), 93.)
        hod.compute_stats.assert_called_once_with('cat', stat=hod._stats_to_fit,
                                                 tracers=hod._tracer_to_fit, verbose=False)


if __name__ == '__main__':
    unittest.main()
