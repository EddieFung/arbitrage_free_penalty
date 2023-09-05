from typing import List
import numpy as np
import jax.numpy as jnp

from model import kalman_filter as kf
# import kalman_filter as kf


def yield_basis(decay_rate: float, maturity: float):
    """The three Nelson-Siegel basis functions for Yields.

    Parameters
    ----------
    decay_rate : float
        lambda.
    maturity : float
        time-to-maturity.

    Returns
    -------
    np.array
        Three basis functions for yields.

    """
    if maturity == 0.:
        return np.array([1., 1., 0.])
    return np.array([
        1,
        (1. - jnp.exp(-decay_rate * maturity)) / decay_rate / maturity,
        (1. - jnp.exp(-decay_rate * maturity)) / decay_rate / maturity - \
            jnp.exp(-decay_rate * maturity)
    ])
            
def forward_basis(decay_rate: float, maturity: float):
    """The three Nelson-Siegel basis functions for forward rates.

    Parameters
    ----------
    decay_rate : float
        lambda.
    maturity : float
        time-to-maturity.

    Returns
    -------
    np.array
        Three basis functions for forward rates.

    """
    return np.array([
        1,
        jnp.exp(-decay_rate * maturity),
        decay_rate * maturity * jnp.exp(-decay_rate * maturity)
    ])

class DynamicNelsonSiegel(kf.OUModel):
    """Dynamic Nelson-Siegel model presented in Diebold and Li (2006).
    
    This is a LGSSM with the discretized OU process as the transition process 
    and the three Nelson-Siegel basis functions as the observation matrix with
    zero observation intercepts. Note that this is a model for yields.
    """
    def __init__(
        self, 
        decay_rate: float,
        maturities: List[float],
        delta_t: float = 1/250
    ) -> None:
        """Instantiate the class.

        Parameters
        ----------
        B : np.array
            Observation intercept, shape = [dim_y].
        H : np.ndarray
            Observation matrix, shape = [dim_y, dim_x].
        maturities : List[float]
            Time-to-maturity.
        delta_t : float, optional
            Time span between two observations. The default is 1/250.

        Returns
        -------
        None
        """
        self.decay_rate = decay_rate
        self.maturities = maturities
        B = np.zeros(len(maturities))
        H = np.array([
            yield_basis(self.decay_rate, m) for m in maturities
        ])
        
        super().__init__(B=B, H=H, delta_t=delta_t)