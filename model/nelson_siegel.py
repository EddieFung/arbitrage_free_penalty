from typing import Tuple, List
import numpy as np
import jax.numpy as jnp

from model import kalman_filter as kf


def yield_basis(decay_rates: Tuple[float, float], maturity: float) -> np.array:
    """The five Nelson-Siegel basis functions for Yields.

    Parameters
    ----------
    decay_rates : Tuple[float, float]
        The two decay rates.
    maturity : float
        Time-to-maturity.

    Returns
    -------
    np.array
        Five basis functions for yields.
    """
    if maturity == 0.:
        return jnp.array([1., 1., 0., 1., 0.])
    return jnp.array([
        1,
        (1. - jnp.exp(-decay_rates[0] * maturity)) / decay_rates[0] / maturity,
        - jnp.exp(-decay_rates[0] * maturity) + \
        (1. - jnp.exp(-decay_rates[0] * maturity)) / decay_rates[0] / maturity,
        (1. - jnp.exp(-decay_rates[1] * maturity)) / decay_rates[1] / maturity,
        - jnp.exp(-decay_rates[1] * maturity) + \
        (1. - jnp.exp(-decay_rates[1] * maturity)) / decay_rates[1] / maturity,
    ])
            
def forward_basis(
    decay_rates: Tuple[float, float], 
    maturity: float
) -> np.array:
    """The five Nelson-Siegel basis functions for forward rates.

    Parameters
    ----------
    decay_rates : Tuple[float, float]
        The two decay rates.
    maturity : float
        Time-to-maturity.

    Returns
    -------
    np.array
        Five basis functions for forward rates.
    """
    return np.array([
        1,
        jnp.exp(-decay_rates[0] * maturity),
        decay_rates[0] * maturity * jnp.exp(-decay_rates[0] * maturity),
        jnp.exp(-decay_rates[1] * maturity),
        decay_rates[1] * maturity * jnp.exp(-decay_rates[1] * maturity)
    ])

class DynamicNelsonSiegel(kf.OUModel):
    """Dynamic Nelson-Siegel model presented in Diebold and Li (2006).
    
    This is a LGSSM with the discretized OU process as the transition process 
    and the three Nelson-Siegel basis functions as the observation matrix with
    zero observation intercepts. Note that this is a model for yields.
    """
    def __init__(
        self, 
        decay_rates: Tuple[float, float],
        maturities: List[float],
        delta_t: float = 1/250
    ) -> None:
        """Instantiate the class.

        Parameters
        ----------
        decay_rates : Tuple[float, float]
            The two decay rates.
        maturities : List[float]
            All Time-to-maturity of interest.
        delta_t : float, optional
            Time span between two observations. The default is 1/250.

        Returns
        -------
        None
        """
        self.decay_rates = decay_rates
        self.maturities = maturities
        B = np.zeros(len(maturities))
        H = np.array([
            yield_basis(self.decay_rates, m) for m in maturities
        ])
        
        super().__init__(B=B, H=H, delta_t=delta_t)
