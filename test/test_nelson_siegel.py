import numpy as np
import unittest

from utils import kalman_filter as kf
from utils import nelson_siegel as ns

class TestNelsonSiegel(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.decay_rates = [0.5, 1.2]
        cls.maturities = [1., 2., 5., 20.]  # randomly selected
    
    def test_three_yield_basis(self):
        for m in self.maturities:
            yield_loadings = ns.three_yield_basis(self.decay_rates[0], m)
            np.testing.assert_array_less(  # check non-negative and shape=[3]
                np.zeros(3) - (1e-8),
                yield_loadings
                )
    def test_yield_basis(self):
        for maturity in self.maturities:
            yield_loadings = ns.yield_basis(self.decay_rates, maturity)
            np.testing.assert_array_less(  # check non-negative and shape=[5]
                np.zeros(5) - (1e-8),
                yield_loadings
                )
            
    def test_forward_basis(self):
        for maturity in self.maturities:
            forward_loadings = ns.forward_basis(self.decay_rates, maturity)
            np.testing.assert_array_less(  # check non-negative and shape=[5]
                np.zeros(5) - (1e-8),
                forward_loadings
                ) 
            
class TestDynamicNelsonSiegel(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.decay_rates = [0.5, 1.2]
        cls.maturities = [1., 2., 5., 20.]  # randomly selected
    
    def test_type(self):
        dns_model = ns.DynamicNelsonSiegel(
            maturities=self.maturities,
            decay_rates=self.decay_rates
        )
        isinstance(dns_model.specify_filter(), kf.OUModel)

if __name__ == '__main__':
    unittest.main()
