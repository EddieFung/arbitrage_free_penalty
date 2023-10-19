import numpy as np
import unittest

import jax.numpy as jnp

from model.afns import adjustment


exact_A = lambda m, rates: m**2 / 6

def exact_B(m: float, rates: np.array):
    return 1 / (2 * rates[0]**2) -  \
        (1 - jnp.exp(-rates[0] * m)) / rates[0]**3 / m + \
        (1 - jnp.exp(-2 * rates[0] * m)) / (4 * rates[0]**3 * m)
        
def exact_C(m: float, rates: np.array):
    return 1 / (2 * rates[1]**2) -  \
        (1 - jnp.exp(-rates[1] * m)) / rates[1]**3 / m + \
        (1 - jnp.exp(-2 * rates[1] * m)) / (4 * rates[1]**3 * m)

def exact_D(m: float, rates: np.array):
    return 1 / (2 * rates[0]**2) + jnp.exp(-rates[0] * m) / rates[0]**2 - \
        m * jnp.exp(- 2 * m * rates[0]) / 4 / rates[0] - \
        3 * jnp.exp(- 2 * m * rates[0]) / 4 / rates[0]**2 - \
        2 * (1 - jnp.exp(-rates[0] * m)) / rates[0]**3 / m + \
        5 * (1 - jnp.exp(-2 * rates[0] * m)) / (8 * rates[0]**3 * m)
        
def exact_E(m: float, rates: np.array):
    return 1 / (2 * rates[1]**2) + jnp.exp(-rates[1] * m) / rates[1]**2 - \
        m * jnp.exp(- 2 * m * rates[1]) / 4 / rates[1] - \
        3 * jnp.exp(- 2 * m * rates[1]) / 4 / rates[1]**2 - \
        2 * (1 - jnp.exp(-rates[1] * m)) / rates[1]**3 / m + \
        5 * (1 - jnp.exp(-2 * rates[1] * m)) / (8 * rates[1]**3 * m)

def exact_F(m: float, rates: np.array):
    return m / (2 * rates[0]) + jnp.exp(-rates[0] * m) / rates[0]**2 - \
        (1 - jnp.exp(-rates[0] * m)) / rates[0]**3 / m
        
def exact_G(m: float, rates: np.array):
    return m / (2 * rates[1]) + jnp.exp(-rates[1] * m) / rates[1]**2 - \
        (1 - jnp.exp(-rates[1] * m)) / rates[1]**3 / m

def exact_H(m: float, rates: np.array):
    return 3 * jnp.exp(-rates[0] * m) / rates[0]**2 + m / 2 / rates[0] + \
        jnp.exp(-rates[0] * m) * m / rates[0] - \
        3 * (1 - jnp.exp(-rates[0] * m)) / m / rates[0]**3

def exact_I(m: float, rates: np.array):
    return 3 * jnp.exp(-rates[1] * m) / rates[1]**2 + m / 2 / rates[1] + \
        jnp.exp(-rates[1] * m) * m / rates[1] - \
        3 * (1 - jnp.exp(-rates[1] * m)) / m / rates[1]**3
        
def exact_J(m: float, rates: np.array):
    return 1 / rates[0] / rates[1] - \
        (1 - jnp.exp(-m * rates[0])) / (rates[0]**2 * rates[1] * m) - \
        (1 - jnp.exp(-m * rates[1])) / (rates[0] * rates[1]**2 * m) + \
        (1 - jnp.exp(-m * jnp.sum(rates))) / (jnp.prod(rates) * jnp.sum(rates) * m)
                    
def exact_K(m: float, rates: np.array):
    return (1 + jnp.exp(-rates[0] * m)) / rates[0]**2 - \
        jnp.exp(-2 * m * rates[0]) / 2 / rates[0]**2 - \
        3 * (1 - jnp.exp(-m * rates[0])) / m / rates[0]**3 + \
        3 * (1 - jnp.exp(-2 * rates[0] * m)) / (4 * rates[0]**3 * m)

def exact_L(m: float, rates: np.array):
    return (1 + jnp.exp(-rates[1] * m)) / jnp.prod(rates) - \
        jnp.exp(-jnp.sum(rates) * m) / jnp.sum(rates) / rates[0] - \
        (1 - jnp.exp(-rates[0] * m)) / (rates[0]**2 * rates[1] * m) - \
        2 * (1 - jnp.exp(-rates[1] * m)) / (rates[0] * rates[1]**2 * m) + \
        (rates[0] + 2 * rates[1]) * (1 - jnp.exp(-jnp.sum(rates) * m)) / \
            (jnp.sum(rates)**2 * jnp.prod(rates) * m)
            
def exact_M(m: float, rates: np.array):
    return (1 + jnp.exp(-rates[0] * m)) / jnp.prod(rates) - \
        jnp.exp(-jnp.sum(rates) * m) / jnp.sum(rates) / rates[1] - \
        (1 - jnp.exp(-rates[1] * m)) / (rates[1]**2 * rates[0] * m) - \
        2 * (1 - jnp.exp(-rates[0] * m)) / (rates[1] * rates[0]**2 * m) + \
        (rates[1] + 2 * rates[0]) * (1 - jnp.exp(-jnp.sum(rates) * m)) / \
            (jnp.sum(rates)**2 * jnp.prod(rates) * m)

def exact_N(m: float, rates: np.array):
    return (1 + jnp.exp(-rates[1] * m)) / rates[1]**2 - \
        jnp.exp(-2 * m * rates[1]) / 2 / rates[1]**2 - \
        3 * (1 - jnp.exp(-m * rates[1])) / m / rates[1]**3 + \
        3 * (1 - jnp.exp(-2 * rates[1] * m)) / (4 * rates[1]**3 * m)

def exact_O(m: float, rates: np.array):
    return (1 + jnp.sum(jnp.exp(-m * rates)))/ (rates[0] * rates[1]) - \
        jnp.sum(1 / rates) / jnp.sum(rates) * jnp.exp(-m * jnp.sum(rates)) - \
        2 * jnp.exp(-m * jnp.sum(rates)) / jnp.sum(rates)**2 - \
        m * jnp.exp(-m * jnp.sum(rates)) / jnp.sum(rates) - \
        2 * (1 - jnp.exp(-m * rates[0])) / (rates[0]**2 * rates[1] * m) - \
        2 * (1 - jnp.exp(-m * rates[1])) / (rates[0] * rates[1]**2 * m) + \
        2 * (1 - jnp.exp(-m * jnp.sum(rates))) / jnp.sum(rates)**3 / m + \
        jnp.sum(1 / rates) / jnp.sum(rates)**2 * \
            (1 - jnp.exp(-m * jnp.sum(rates))) / m + \
        (1 - jnp.exp(-m * jnp.sum(rates))) / (jnp.prod(rates) * jnp.sum(rates) * m)

class TestAdjustment(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.m = 1.5
        cls.rates = np.array([0.5, 0.75])
        cls.rates_2 = np.array([0.5, 0.5])
        cls.rates_3 = np.array([0.75, 0.75])
        cls.rates_4 = np.array([0.75, 0.5])
        
    def test_dependent_5_factor(self):
        """Compare adjustment terms in dependent 5-factor case.
        """
        low_tri_mat = np.array([
            [0.01, 0, 0, 0, 0],
            [0.011, 0.012, 0, 0, 0],
            [0.013, 0.014, 0.015, 0, 0],
            [0.016, 0.017, 0.018, 0.019, 0],
            [0.020, 0.021, 0.022, 0.023, 0.024]
        ])
        cov_mat = jnp.matmul(low_tri_mat, low_tri_mat.T)
        ans = exact_A(self.m, self.rates) * cov_mat[0, 0]+ \
            exact_B(self.m, self.rates) * cov_mat[1, 1] + \
            exact_C(self.m, self.rates) * cov_mat[2, 2]+ \
            exact_D(self.m, self.rates) * cov_mat[3, 3] + \
            exact_E(self.m, self.rates) * cov_mat[4, 4] + \
            exact_F(self.m, self.rates) * low_tri_mat[0, 0] * low_tri_mat[1, 0] + \
            exact_G(self.m, self.rates) * low_tri_mat[0, 0] * low_tri_mat[2, 0] + \
            exact_H(self.m, self.rates) * low_tri_mat[0, 0] * low_tri_mat[3, 0] + \
            exact_I(self.m, self.rates) * low_tri_mat[0, 0] * low_tri_mat[4, 0] + \
            exact_J(self.m, self.rates) * (
                low_tri_mat[1, 0] * low_tri_mat[2, 0] + \
                low_tri_mat[1, 1] * low_tri_mat[2, 1]) + \
            exact_K(self.m, self.rates) * (
                low_tri_mat[1, 0] * low_tri_mat[3, 0] + \
                low_tri_mat[1, 1] * low_tri_mat[3, 1]) + \
            exact_L(self.m, self.rates) * (
                low_tri_mat[1, 0] * low_tri_mat[4, 0] + \
                low_tri_mat[1, 1] * low_tri_mat[4, 1]) + \
            exact_M(self.m, self.rates) * (
                low_tri_mat[2, 0] * low_tri_mat[3, 0] + \
                low_tri_mat[2, 1] * low_tri_mat[3, 1] + \
                low_tri_mat[2, 2] * low_tri_mat[3, 2]) + \
            exact_N(self.m, self.rates) * (
                low_tri_mat[2, 0] * low_tri_mat[4, 0] + \
                low_tri_mat[2, 1] * low_tri_mat[4, 1] + \
                low_tri_mat[2, 2] * low_tri_mat[4, 2]) + \
            exact_O(self.m, self.rates) * (
                low_tri_mat[3, 0] * low_tri_mat[4, 0] + \
                low_tri_mat[3, 1] * low_tri_mat[4, 1] + \
                low_tri_mat[3, 2] * low_tri_mat[4, 2] + \
                low_tri_mat[3, 3] * low_tri_mat[4, 3]) 
            
        cov_mat = jnp.matmul(low_tri_mat, low_tri_mat.T)
        mat = adjustment.adjustment_matrix(self.m, self.rates)
        self.assertAlmostEqual(
            jnp.sum(jnp.diag(jnp.matmul(cov_mat, -mat))),
            -ans,
            6
        )

    def test_independent_5_factor(self):
        """Compare adjustment terms in independent 5-factor case.
        """
        cov_mat_diag = np.random.uniform(0.01, 0.1, 5)
        cov_mat = np.diag(cov_mat_diag**2)
        mat = adjustment.adjustment_matrix(self.m, self.rates)
        self.assertAlmostEqual(
            jnp.sum(jnp.diag(jnp.matmul(cov_mat, -mat))),
            -jnp.sum(cov_mat_diag**2 * np.array([
                exact_A(self.m, self.rates), 
                exact_B(self.m, self.rates),
                exact_C(self.m, self.rates),
                exact_D(self.m, self.rates),
                exact_E(self.m, self.rates)
            ])),
            6
        )
        
    def test_adjustment_matrix_shape(self):
        np.testing.assert_equal(
            adjustment.adjustment_matrix(0., np.array([0.5, 0.5])).shape,
            (5, 5)
        )
        
        np.testing.assert_equal(
            adjustment.adjustment_matrix(self.m, np.array([0.5, 0.5])).shape,
            (5, 5)
        )
        
        np.testing.assert_equal(
            adjustment.adjustment_matrix(self.m, np.array([0.5, 0.75])).shape,
            (5, 5)
        )
        
    def test_fn_1(self):
        """Term F, G in Christensen et al. (2009)"""
        np.testing.assert_array_equal(
            2 * adjustment.fn_1(self.m, self.rates),
            (exact_F(self.m, self.rates), exact_G(self.m, self.rates))
        )
        
    def test_fn_2(self):
        """Term H, I in Christensen et al. (2009)"""
        np.testing.assert_array_equal(
            2 * adjustment.fn_2(self.m, self.rates),
            (exact_H(self.m, self.rates), exact_I(self.m, self.rates))
        )
        
    def test_fn_3_same_rate(self):
        """Term B, C in Christensen et al. (2009)"""
        self.assertAlmostEqual(
            adjustment.fn_3(self.m, self.rates_2),
            exact_B(self.m, self.rates),
            6
        )
        
        self.assertAlmostEqual(
            adjustment.fn_3(self.m, self.rates_3),
            exact_C(self.m, self.rates),
            6
        )
        
    def test_fn_3_diff_rate(self):
        """Term J in Christensen et al. (2009)"""
        self.assertAlmostEqual(
            2 * adjustment.fn_3(self.m, self.rates),
            exact_J(self.m, self.rates),
            6
        )
        
        self.assertAlmostEqual(
            2 * adjustment.fn_3(self.m, self.rates_4),
            exact_J(self.m, self.rates),
            6
        )
    
    def test_fn_4_same_rate(self):
        """Term D, E in Christensen et al. (2009)"""
        self.assertAlmostEqual(
            adjustment.fn_4(self.m, self.rates_2),
            exact_D(self.m, self.rates),
            6
        )
        self.assertAlmostEqual(
            adjustment.fn_4(self.m, self.rates_3),
            exact_E(self.m, self.rates),
            6
        )
        
    def test_fn_4_diff_rate(self):
        """Term O in Christensen et al. (2009)"""
        self.assertAlmostEqual(
            2 * adjustment.fn_4(self.m, self.rates),
            exact_O(self.m, self.rates),
            6
        )
        
        self.assertAlmostEqual(
            2 * adjustment.fn_4(self.m, self.rates_4),
            exact_O(self.m, self.rates),
            6
        )
        
    def test_fn_5_same_rate(self):
        """Term K, N in Christensen et al. (2009)"""
        self.assertAlmostEqual(
            2 * adjustment.fn_5(self.m, self.rates_2),
            exact_K(self.m, self.rates),
            6
        )
        
        self.assertAlmostEqual(
            2 * adjustment.fn_5(self.m, self.rates_3),
            exact_N(self.m, self.rates),
            6
        )
        
    def test_fn_5_diff_rate(self):
        """Term L, M in Christensen et al. (2009)"""
        self.assertAlmostEqual(
            2 * adjustment.fn_5(self.m, self.rates),
            exact_L(self.m, self.rates),
            6
        )
        
        self.assertAlmostEqual(
            2 * adjustment.fn_5(self.m, self.rates_4),
            exact_M(self.m, self.rates),
            6
        )

        
if __name__ == '__main__':
    unittest.main()
