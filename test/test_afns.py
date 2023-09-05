import numpy as np
import unittest

from model import afns
from model import kalman_filter as kf

class Test_yield_adjustment(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.decay_rate = 0.5
        cls.maturities = [1., 2., 5., 20.]  # randomly selected
    
    def test_afns_yield_adjustment(self):
        sigma = np.array([0.0051, 0.0110, 0.0264]) # from Christensen et al. (2011)
        for maturity in self.maturities:
            self.assertLess(
                afns.arbitrage_free_yield_adjustment(self.decay_rate, sigma, maturity),
                0.
            )
            
class TestAFNS(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        np.random.seed(0)
        
        cls.maturities = [1., 2., 5., 20.]  # randomly selected
        cls.dim_y = len(cls.maturities)
        cls.dim_x = 3
        
        cls.model = afns.AFNS(cls.maturities)     
        
    def test_type(self):
        isinstance(self.model.specify_filter(), kf.BaseLGSSM)
        
    def test_initialize(self):
        np.random.seed(3)
        
        df = np.random.normal(size=(10, self.dim_y))
        self.model.initialize(df)
        
        np.testing.assert_equal(len(self.model._log_k), 3)
        np.testing.assert_equal(len(self.model._theta), 3) 
        np.testing.assert_equal(len(self.model._log_sigma), 3) 
        np.testing.assert_equal(len(self.model._log_obs_sd), self.dim_y) 
            
if __name__ == '__main__':
    unittest.main()
