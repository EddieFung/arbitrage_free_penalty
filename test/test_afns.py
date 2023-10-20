import numpy as np
import unittest

from model.afns import afns
from utils import kalman_filter as kf


class TestAFNS(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        np.random.seed(0)
        
        cls.maturities = [1., 2., 5., 7., 10., 20.]  # randomly selected
        cls.dim_y = len(cls.maturities)
        cls.dim_x = 5
        
        cls.model = afns.AFNS(cls.maturities)     
        
    def test_type(self):
        isinstance(self.model.specify_filter(), kf.BaseLGSSM)

    def test_initialize(self):
        np.random.seed(3)
        
        df = np.random.normal(size=(10, self.dim_y))
        self.model.initialize(df)
        
        np.testing.assert_equal(self.model._k_p.shape, [5,5])
        np.testing.assert_equal(len(self.model._theta_p), 5) 
        np.testing.assert_equal(len(self.model._log_sd), 5) 
        np.testing.assert_equal(len(self.model._log_obs_sd), self.dim_y) 
            
    def test_specify_adjustments(self):
        adjustments = self.model.specify_adjustments()
        np.testing.assert_array_less(
            adjustments, np.zeros(self.dim_y)
        )

if __name__ == '__main__':
    unittest.main()
