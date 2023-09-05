from typing import List
import numpy as np

import jax
import jax.numpy as jnp

from model import kalman_filter as kf
from model import nelson_siegel as ns

def arbitrage_free_yield_adjustment(
        decay_rate: float, 
        sigma: float, 
        maturity: float
        ):
    """The yield adjustment term of independent 3-factor AFNS.

    Parameters
    ----------
    decay_rate : float
        lambda.
    sigma : float
        standard deviation in continuous-time transition equation
    maturity : float
        time-to-maturity.

    Returns
    ------
    float
        yield adjustment term.

    """
    if maturity == 0.:
        return np.array(0.)
    return - (sigma[0] * maturity)**2 / 6. - sigma[1]**2 * (
        0.5 / decay_rate**2 - 
        (1 - jnp.exp(-decay_rate * maturity)) / (decay_rate**3 * maturity) +
        (1. - jnp.exp(-2 * decay_rate * maturity)) / (4. * decay_rate**3 * maturity)
    ) - sigma[2]**2 * (
        0.5 / decay_rate**2 + jnp.exp(-decay_rate * maturity) / decay_rate**2 - 
        maturity * jnp.exp(-2 * decay_rate * maturity) / (4. * decay_rate) - 
        3. * jnp.exp(-2 * decay_rate * maturity) / (4. * decay_rate**2) - 
        2. * (1. - jnp.exp(-decay_rate * maturity)) / (decay_rate**3 * maturity) +
        5. * (1. - jnp.exp(-2. * decay_rate * maturity)) / (8. * decay_rate**3 * maturity)
    )
        
class AFNS(kf.OUTransitionModel):
    
    def __init__(
        self, 
        maturities: List[float],
        delta_t: float = 1/250
    ) -> None:
        super().__init__(delta_t=delta_t)
        self._log_rate = jnp.log(0.713131)
        self.maturities = maturities
        self._log_obs_sd = np.zeros(len(maturities))
        
    def specify_filter(self) -> kf.BaseLGSSM:
        """Specify the LGSSM given the parameter values. 

        Returns
        -------
        BaseLGSSM
            The LGSSM. 
        """
        return self._specify_filter([
            self._log_rate, self._log_k, self._theta, self._log_sigma, self._log_obs_sd
        ])
        
    def _specify_filter(self, pars: List) -> kf.BaseLGSSM:
        """Hidden method to specify the LGSSM.
        
        Parameters
        ----------
        pars : List
            Parameter values.

        Returns
        -------
        BaseLGSSM
            The LGSSM. 
        """
        log_rate, log_k, theta, log_sigma, log_obs_sd = pars
        (hat_A, hat_F, hat_Q), hat_R, (hat_m0, hat_P0) = super()._specify_transition([log_k, theta, log_sigma, log_obs_sd])
        
        hat_B = np.array([arbitrage_free_yield_adjustment(
            jnp.exp(log_rate), jnp.exp(log_sigma), m
        ) for m in self.maturities])
        hat_H = np.array([ns.yield_basis(jnp.exp(log_rate), m) for m in self.maturities])
        
        return kf.BaseLGSSM(hat_A, hat_F, hat_Q, hat_B, hat_H, hat_R, hat_m0, hat_P0)
    
    def initialize(self, df: np.ndarray) -> None:
        """Initialize parameters given df.

        Parameters
        ----------
        df : np.ndarray
            Data, shape = [dim_t, dim_y]

        Returns
        -------
        None
        """
        hat_H = np.array([ns.yield_basis(jnp.exp(self._log_rate), m) for m in self.maturities])
        super()._initialize(df, hat_H)

    def inference(self, df: np.ndarray, iterations: int = 3) -> None:
        """Perform inference on df.
        
        Update parameters using Adam optimizer on -ve log-likelihood.

        Parameters
        ----------
        df : np.ndarray
            Data, shape = [dim_t, dim_y]
        iterations : int, optional
            Number of iterations of Adam optimizer. The default is 3.

        Returns
        -------
        None
        """
        @jax.jit
        def neg_log_like(pars, df):
            model = self._specify_filter(pars)
            return -jnp.mean(model.forward_filter(df)[2])
        
        self.initialize(df)
        pars = [self._log_rate, self._log_k, self._theta, self._log_sigma, self._log_obs_sd]
        pars = super()._inference(pars, df, neg_log_like, iterations)
        self._log_rate, self._log_k, self._theta, self._log_sigma, self._log_obs_sd = pars
