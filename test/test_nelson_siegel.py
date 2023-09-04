import sys
sys.path.append('..')

import numpy as np
import model.nelson_siegel as ns
import unittest


class TestNelsonSiegel(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.decay_rate = 0.5
        cls.maturities = [1., 2., 5., 20.]  # randomly selected
    
    def test_yield_basis(self):
        for maturity in self.maturities:
            yield_loadings = ns.yield_basis(self.decay_rate, maturity)
            np.testing.assert_array_less(  # check non-negative and shape=[3]
                np.zeros(3) - (1e-8),
                yield_loadings
                )
    
    def test_afns_yield_adjustment(self):
        sigma = np.array([0.0051, 0.0110, 0.0264]) # from Christensen et al. (2011)
        for maturity in self.maturities:
            self.assertLess(
                ns.arbitrage_free_yield_adjustment(self.decay_rate, sigma, maturity),
                0.
            )
            
    def test_forward_basis(self):
        for maturity in self.maturities:
            forward_loadings = ns.forward_basis(self.decay_rate, maturity)
            np.testing.assert_array_less(  # check non-negative and shape=[3]
                np.zeros(3) - (1e-8),
                forward_loadings
                ) 
            
            
if __name__ == '__main__':
    unittest.main()
