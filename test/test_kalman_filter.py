import numpy as np
import unittest

from model import kalman_filter as kf


class TestBaseLGSSM(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        np.random.seed(0)
        
        cls.dim_x = 3
        cls.dim_y = 5
        
        cls.A = np.random.normal(size=cls.dim_x) * 0.1
        cls.F = np.diag(np.random.uniform(low=0.5, high=0.9, size=cls.dim_x))
        cls.Q = np.diag(np.random.uniform(low=0.8, high=1.3, size=cls.dim_x))
        cls.B = np.random.normal(size=cls.dim_y) * 0.2
        cls.H = np.random.normal(size=(cls.dim_y, cls.dim_x)) * 0.5
        cls.R = np.diag(np.random.uniform(low=0.1, high=0.3, size=cls.dim_y))
        cls.m0 = np.zeros(cls.dim_x)
        cls.P0 = np.eye(cls.dim_x)
        
        cls.model = kf.BaseLGSSM(
            cls.A,
            cls.F, 
            cls.Q, 
            cls.B, 
            cls.H, 
            cls.R,
            cls.m0,
            cls.P0
        )
        
    def exact_one_step_filter(self, m , P, observation):
        res = observation - self.B
        inv_R = np.linalg.inv(self.R)
        exact_P = np.linalg.inv(
            np.linalg.inv(P) + self.H.T @ inv_R @ self.H
        )
        exact_m = (res @ inv_R @ self.H + m @ np.linalg.inv(P)) @ exact_P
        return exact_m, exact_P
        
    def test_one_step_filter(self):
        np.random.seed(3)
        
        one_observation = np.random.normal(size=self.dim_y)
        _, (est_m, est_P, _) = self.model.one_step_filter(
            (self.m0, self.P0, 0), one_observation
        )
        exact_m, exact_P = self.exact_one_step_filter(
            self.A + self.F @ self.m0, 
            self.F @ self.P0 @ self.F.T + self.Q,
            one_observation
        )
        
        np.testing.assert_array_almost_equal(
            est_P,
            exact_P
        )
        
        np.testing.assert_array_almost_equal(
            est_m,
            exact_m
        )
        
    
    def test_forward_filter(self):
        np.random.seed(3)
        
        two_observation = np.random.normal(size=(2, self.dim_y))
        
        one_m, one_P = self.exact_one_step_filter(
            self.A + self.F @ self.m0, 
            self.F @ self.P0 @ self.F.T + self.Q,
            two_observation[0, :]
        )
        two_m, two_P = self.exact_one_step_filter(
            self.A + self.F @ one_m, 
            self.F @ one_P @ self.F.T + self.Q,
            two_observation[1, :]
        )
        
        m, P, likeli = self.model.forward_filter(two_observation)
        
        np.testing.assert_array_almost_equal(
            two_m,
            m[-1, :]
        )
        
        np.testing.assert_array_almost_equal(
            two_P,
            P[-1, :]
        )
        
class TestOUTransitionModel(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        np.random.seed(0)
        
        cls.dim_x = 3
        cls.dim_y = 5
        cls.delta_t = 1/250
        
        cls.B = np.random.normal(size=cls.dim_y) * 0.2
        cls.H = np.random.normal(size=(cls.dim_y, cls.dim_x)) * 0.5
        
        cls.model = kf.OUTransitionModel(cls.delta_t)   
        
    def test_initialize(self):
        np.random.seed(3)
        
        df = np.random.normal(size=(10, self.dim_y))
        self.model._initialize(df, self.H)
        
        np.testing.assert_equal(len(self.model._log_k), 3)
        np.testing.assert_equal(len(self.model._theta), 3) 
        np.testing.assert_equal(len(self.model._log_sigma), 3) 
        np.testing.assert_equal(len(self.model._log_obs_sd), self.dim_y) 
        
    def test_specify_transition(self):
        a, b, c = self.model._specify_transition([
            self.model._log_k, self.model._theta, self.model._log_sigma, np.zeros(self.dim_y)
        ])
        np.testing.assert_equal(len(a), 3)
        np.testing.assert_equal(len(b), self.dim_y)
        np.testing.assert_equal(len(c), 2)
        
        
class TestOUModel(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        np.random.seed(0)
        
        cls.dim_x = 3
        cls.dim_y = 5
        
        cls.B = np.random.normal(size=cls.dim_y) * 0.2
        cls.H = np.random.normal(size=(cls.dim_y, cls.dim_x)) * 0.5
        
        cls.model = kf.OUModel(cls.B, cls.H)     
        
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
