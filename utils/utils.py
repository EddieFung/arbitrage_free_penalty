from typing import Callable
import numpy as np
import jax.numpy as jnp


class Grids:
    """Define a fine grid on the maturity dimension for AF_loss.
    """
    
    def __init__(self, start: float = 0, stop: float = 30, num: int = 500):
        """Instantiate the class.
        
        Girds of uniform length are generated. The first grid and the last grid
        are (start, (stop-start)/(num-1)) and ((stop-start)/(num-1), stop). The 
        end-point of each grid is saved.
        
        Parameters
        ----------
        start : float, optional
            The start value of the grid. The default is 0.
        stop : float, optional
            The end value of the grid. The default is 30.
        num : int, optional
            Number of grids + 1. The default is 500.

        Returns
        -------
        None.
        """
        self.start = start
        self.stop = stop
        self.num = num
        linspace = np.linspace(start, stop, num)

        self.stepsize = linspace[1] - linspace[0]
        self.grids = linspace[1:]

    def integrate(self, fn: Callable[[float], float]):
        """Integrate a function using the mid-point rule.

        Parameters
        ----------
        fn : Callable[[float], float]
            A callable object.

        Returns
        -------
        np.array
            The integrals of fn with different upper limit.
        """
        full_grid = fn(self.grids - self.stepsize / 2)
        return jnp.cumsum(full_grid * self.stepsize, 0)  # size = [self.num-1]
