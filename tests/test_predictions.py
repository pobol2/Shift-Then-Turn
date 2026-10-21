"""The closed-form predictions against their limits and the values quoted in the paper."""

import unittest

import numpy as np

from simulation.predictions import K0, first_order_slip, relax, slip


class Shift(unittest.TestCase):
    def test_tilt_is_linear_in_offset_and_below_asymptote(self):
        # Slopes of 1.59 and 2.28 deg/mm from Eq. (relax_end), Section 8.
        for lam, slope in ((37.10795, 1.59), (30.0, 2.28)):
            with self.subTest(lam=lam):
                end = relax(lam, 1.0)[1][-1]
                self.assertAlmostEqual(end, slope, delta=0.005)
                self.assertEqual(relax(lam, 0.0)[1][-1], 0.0)
                self.assertAlmostEqual(relax(lam, -0.5)[1][-1], -0.5 * end)
                self.assertLess(end, np.degrees(1 / (K0 - lam)))


class Turn(unittest.TestCase):
    def test_integrated_slip_tends_to_first_order(self):
        self.assertEqual(slip(0, 16), 0.0)
        for rho in (0.25, 0.5):
            with self.subTest(rho=rho):
                self.assertAlmostEqual(slip(rho, 16) / first_order_slip(rho, 16), 1, delta=0.01)

    def test_first_order_overstates_large_pads(self):
        for rho in (1, 2, 4):
            with self.subTest(rho=rho):
                self.assertLess(slip(rho, 16), first_order_slip(rho, 16))


if __name__ == '__main__':
    unittest.main()
