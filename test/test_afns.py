import numpy as np
import unittest

from model.afns import afns
from utils import kalman_filter as kf


class TestAFINS(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        np.random.seed(0)
        
        cls.maturities = [1., 2., 5., 7., 10., 20.]  # randomly selected
        cls.dim_y = len(cls.maturities)
        cls.dim_x = 3
        
        cls.model = afns.AFINS(cls.maturities)     
        
    def test_type(self):
        isinstance(self.model.specify_filter(), kf.BaseLGSSM)

    def test_initialize(self):
        np.random.seed(3)
        
        df = np.random.normal(size=(10, self.dim_y))
        self.model.initialize(df)
        
        np.testing.assert_equal(self.model._k_p.shape, [self.dim_x, self.dim_x])
        np.testing.assert_equal(len(self.model._theta_p), self.dim_x) 
        np.testing.assert_equal(len(self.model._log_sd), self.dim_x) 
        np.testing.assert_equal(len(self.model._log_obs_sd), self.dim_y) 
            
    def test_specify_adjustments(self):
        adjustments = self.model.specify_adjustments()
        np.testing.assert_array_less(
            adjustments, np.zeros(self.dim_y)
        )
        
    def test_observation_components(self):
        cov_mat = self.model._independent_continuous_covariance(self.model._log_sd)
        B, H = self.model._observation_components(
            cov_mat, (self.model._log_rate, self.model._k_p)
        )
        
        np.testing.assert_equal(len(B), self.dim_y)
        np.testing.assert_equal(H.shape, (self.dim_y, self.dim_x))
      

class TestAFNS(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        np.random.seed(0)
        
        cls.maturities = [1., 2., 5., 7., 10., 20.]  # randomly selected
        cls.dim_y = len(cls.maturities)
        cls.dim_x = 3
        
        cls.model = afns.AFNS(cls.maturities)     
        
    def test_type(self):
        isinstance(self.model.specify_filter(), kf.BaseLGSSM)

    def test_specify_adjustments(self):
        adjustments = self.model.specify_adjustments()
        np.testing.assert_array_less(
            adjustments, np.zeros(self.dim_y)
        )


if __name__ == '__main__':
    unittest.main()
