from typing import Tuple, List
import numpy as np

import jax
import jax.numpy as jnp

from model.afns import adjustment
from utils import kalman_filter as kf
from utils import nelson_siegel as ns

        
class AFNS(kf.OUTransitionModel):
    """Arbitrage-free Nelson-Siegel yield model from Christensen et al. (2011).
    """
    def __init__(
        self, 
        maturities: List[float],
        delta_t: float = 1/250,
        decay_rates: Tuple[float, float] = [0.4, 1.2]
    ) -> None:
        """Instantiate the class.
        
        Parameters
        ----------
        maturities : List[float]
            All Time-to-maturity of interest.
        delta_t : float, optional
            Time span between two observations. The default is 1/250.

        Returns
        -------
        None
        """
        super().__init__(delta_t=delta_t)
        self.maturities = maturities
        self._log_rates = np.log(decay_rates)
        self._log_obs_sd = np.zeros(len(maturities))
        
    def specify_adjustments(self) -> float:
        """Specify the yield adjustment term of AFNS.

        Returns
        -------
        float
            yield adjustment term.

        """
        return self._specify_adjustments(
            pars=[self._log_rates, self._k_p, self._log_sd, self._transformed_corr]
        )
    
    def _specify_adjustments(self, pars: Tuple) -> float:
        """Hidden method to specify the yield adjustment term of AFNS.

        Returns
        -------
        float
            yield adjustment term.

        """
        log_rates, k_p, log_sd, transformed_corr = pars
        rates = jnp.exp(log_rates)
        cov_mat, eig_val, eig_vector = super()._sepcify_continuous_dynamic([
            k_p, log_sd, transformed_corr
        ])
        adjustments = np.zeros(len(self.maturities))
        for idx, m in enumerate(self.maturities):
            adjustment_mat = adjustment.adjustment_matrix(m, rates)
            adjustments[idx] = jnp.sum(jnp.diag(
                jnp.matmul(cov_mat, -adjustment_mat)
            ))
            
        return adjustments
        
    
    def specify_filter(self) -> kf.BaseLGSSM:
        """Specify the LGSSM given the parameter values. 

        Returns
        -------
        kf.BaseLGSSM
            The LGSSM. 
        """
        return self._specify_filter([
            self._log_rates, self._k_p, self._theta_p, self._log_sd, 
            self._transformed_corr, self._log_obs_sd
        ])
        
    def _specify_filter(self, pars: Tuple) -> kf.BaseLGSSM:
        """Hidden method to specify the LGSSM.
        
        Parameters
        ----------
        pars : Tuple
            Parameter values.

        Returns
        -------
        kf.BaseLGSSM
            The LGSSM. 
        """
        log_rates, k_p, theta_p, log_sd, transformed_corr, log_obs_sd = pars
        (hat_A, hat_F, hat_Q), hat_R, (hat_m0, hat_P0) = super()._sepcify_discrete_dynamic([
            k_p, theta_p, log_sd, transformed_corr, log_obs_sd
        ])
        
        hat_B = self._specify_adjustments(
            [log_rates, k_p, log_sd, transformed_corr]
        )
        hat_H = jnp.array([ns.yield_basis(jnp.exp(log_rates), m) 
                           for m in self.maturities])
        
        return kf.BaseLGSSM(
            hat_A, hat_F, hat_Q, hat_B, hat_H, hat_R, hat_m0, hat_P0
        )
    
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
        hat_H = np.array([ns.yield_basis(jnp.exp(self._log_rates), m) 
                          for m in self.maturities])
        super()._initialize(df, hat_H)

    def inference(
        self, 
        df: np.ndarray, 
        iterations: int = 3, 
        initialized: bool = False
    ) -> None:
        """Perform inference on df.
        
        Update parameters using Adam optimizer on -ve log-likelihood.

        Parameters
        ----------
        df : np.ndarray
            Data, shape = [dim_t, dim_y]
        iterations : int, optional
            Number of iterations of Adam optimizer. The default is 3.
        initialized : bool, optional
            Whether parameters have been initialized. The default is False.

        Returns
        -------
        None
        """
        @jax.jit
        def neg_log_like(pars, df):
            model = self._specify_filter(pars)
            return -jnp.mean(model.forward_filter(df)[2])
        
        if not initialized:
            self.initialize(df)
        pars = [self._log_rates, self._k_p, self._theta_p, self._log_sd, 
                self._transformed_corr, self._log_obs_sd]
        pars = super()._inference(pars, df, neg_log_like, iterations)
        [self._log_rates, self._k_p, self._theta_p, self._log_sd, 
         self._transformed_corr, self._log_obs_sd] = pars
