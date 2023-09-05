import numpy as np
import unittest

from model import utils

class TestGrids(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.start = 0
        cls.stop = 30
        cls.num = 1000
        
        cls.grids = utils.Grids(cls.start, cls.stop, cls.num)
        
    def test_grids_shape(self):
        np.testing.assert_equal(
            self.num - 1,
            len(self.grids.grids)
            ) 
        
    def test_integrate(self):
        """Compare it with exact integration.
        
        1 - e^{-lam x} = \int_0^x lam * e^{-lam s}ds
        """
        lam = 0.5
        position_idx = [0, 3, 19, 75]  # randomly selected points
        
        grids_ans = self.grids.integrate(lambda x: lam * np.exp(-lam * x))
        for idx in position_idx:
            np.testing.assert_almost_equal(
                grids_ans[idx],
                1 - np.exp(-lam * self.grids.grids[idx]),
                5
            )

            
if __name__ == '__main__':
    unittest.main()
