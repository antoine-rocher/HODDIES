"""ALL requests use the same estimator routing as the default statistics."""

import unittest
from unittest.mock import Mock

import numpy as np

from HODDIES import HOD


class TestComputeStatsAll(unittest.TestCase):
    def setUp(self):
        self.hod = HOD.__new__(HOD)
        self.cat = object()
        self.hod.field_particles = object()
        self.hod.args = {'clustering_settings': {
            'xi_rppi': {'pimax': 40},
            'xi_smu': {'multipole_index': [4, 0, 2]},
        }}

        def estimator(*, return_sep, ells=None, pimax=None):
            self.assertTrue(return_sep)
            x = np.array([1., 2.])
            if ells is not None:
                return x, np.array([[ell, ell + 1] for ell in ells])
            if pimax is not None:
                return x, np.array([3., 4.])
            return x, x, np.ones((2, 2))

        self.hod.get_cross_twopoint = Mock(return_value={'A_A': estimator})
        for name in ['get_delta_sigma', 'get_CIC', 'get_cross_PS']:
            setattr(self.hod, name, Mock(return_value={'A_A': ([1., 2.], [3., 4.])}))

    def test_all_aliases_compute_every_source_once(self):
        for request in ['all', 'ALL', 'aLl', ['all'], ('ALL',), ['wp', 'ALL']]:
            with self.subTest(request=request):
                self.hod.get_cross_twopoint.reset_mock()
                for name in ['get_delta_sigma', 'get_CIC', 'get_cross_PS']:
                    getattr(self.hod, name).reset_mock()
                result = self.hod.compute_stats(self.cat, stat=request, tracers=['A'], verbose=True)
                self.assertEqual(set(result), {'wp', 'xi_rppi', 'xi_smu', 'xi_ells',
                                               'delta_sigma', 'CIC', 'power_spectrum'})
                self.assertEqual(self.hod.get_cross_twopoint.call_count, 2)
                for call, mode in zip(self.hod.get_cross_twopoint.call_args_list, ['rppi', 'smu']):
                    self.assertEqual(call[0], (self.cat, mode))
                    self.assertEqual(call[1], {'tracers': ['A'], 'verbose': True,
                                                   'return_pycorr_obj': True})
                for name in ['get_delta_sigma', 'get_CIC']:
                    getattr(self.hod, name).assert_called_once_with(
                        self.cat, tracers=['A'], verbose=True, return_dic=True)
                self.hod.get_cross_PS.assert_called_once_with(self.cat, tracers=['A'], verbose=True)
                # Grouped correlation multipoles retain the configured order.
                np.testing.assert_array_equal(result['xi_ells']['A_A'][1],
                                              [[4, 5], [0, 1], [2, 3]])

    def test_all_matches_default_result(self):
        default = self.hod.compute_stats(self.cat)
        result = self.hod.compute_stats(self.cat, stat='all')
        self.assertEqual(list(result), list(default))
        for stat in default:
            for actual, expected in zip(result[stat]['A_A'], default[stat]['A_A']):
                np.testing.assert_array_equal(actual, expected)

    def test_unknown_name_with_all_still_raises_before_computing(self):
        with self.assertRaises(ValueError):
            self.hod.compute_stats(self.cat, stat=['all', 'unknown'])
        self.hod.get_cross_twopoint.assert_not_called()

    def test_all_preserves_missing_particle_error(self):
        self.hod.field_particles = None
        with self.assertRaisesRegex(ValueError, 'Field particles not set'):
            self.hod.compute_stats(self.cat, stat='ALL')


if __name__ == '__main__':
    unittest.main()
