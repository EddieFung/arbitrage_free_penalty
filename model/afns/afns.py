from typing import Tuple, List
import numpy as np

import jax.numpy as jnp

from model.afns import adjustment
from utils import kalman_filter as kf
from utils import nelson_siegel as ns

    
class AFNS(kf.OUTransitionModel):
    """Arbitrage-free dependent Nelson-Siegel yield model.
    
    This is the dependent 3-factor model from Christensen et al. (2011).
    """
    def __init__(
        self, 
        maturities: List[float],
        delta_t: float = 1/250,
        decay_rate: float = 0.8
    ) -> None:
        """Instantiate the class.
        
        Parameters
        ----------
        maturities : List[float]
            All Time-to-maturity of interest.
        delta_t : float, optional
            Time span between two observations. The default is 1/250.
        decay_rates : float
            The decay rate.

        Returns
        -------
        None
        """
        super().__init__(dim_x=3, delta_t=delta_t)
        self.maturities = maturities
        self._log_rate = np.log(decay_rate)
        self._log_obs_sd = np.zeros(len(maturities))
        
    def specify_adjustments(self) -> float:
        """Specify the yield adjustment term of AFNS.

        Parameters
        ----------
        cov_mat: np.ndarray
            Continuous-time covariance matrix, shape = [dim_x, dim_x].
        
        Returns
        -------
        float
            Yield adjustment terms.
        """
        cov_mat = super()._dependent_continuous_covariance(
            self._log_sd, self._transformed_corr
        )
        return self._specify_adjustments(cov_mat, jnp.exp(self._log_rate))
    
    def _specify_adjustments(self, cov_mat: np.ndarray, decay_rate: float) -> float:
        """Hidden method to specify the yield adjustment term of AFNS.

        Parameters
        ----------
        cov_mat: np.ndarray
            Continuous-time covariance matrix, shape = [dim_x, dim_x].
        decay_rate : float
            The decay rate.

        Returns
        -------
        float
            Yield adjustment terms.
        """
        vectorize_adjustment = lambda m: jnp.sum(jnp.diag(
            jnp.matmul(cov_mat, -adjustment.smaller_adjustment_matrix(m, decay_rate))
        ))
        return jnp.array([vectorize_adjustment(m) for m in self.maturities])
    
    def _observation_components(self, cov_mat: np.ndarray, decay_rate: float) -> Tuple:
        hat_B = self._specify_adjustments(cov_mat, decay_rate)
        hat_H = jnp.array([ns.three_yield_basis(decay_rate, m) 
                           for m in self.maturities])
        return hat_B, hat_H
    
    def specify_filter(self) -> kf.BaseLGSSM:
        """Specify the LGSSM given the parameter values. 

        Returns
        -------
        kf.BaseLGSSM
            The LGSSM. 
        """
        return self._specify_filter([
            self._log_rate, self._k_p, self._theta_p, self._log_sd, 
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
        log_rate, k_p, theta_p, log_sd, transformed_corr, log_obs_sd = pars
        cov_mat = super()._dependent_continuous_covariance(
            log_sd, transformed_corr
        )
        (hat_A, hat_F, hat_Q), hat_R, (hat_m0, hat_P0) = super()._discrete_components(
            cov_mat, (k_p, theta_p, log_obs_sd)
        )
        
        hat_B, hat_H = self._observation_components(cov_mat, jnp.exp(log_rate))
        
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
        hat_H = np.array([ns.three_yield_basis(jnp.exp(self._log_rate), m) 
                          for m in self.maturities])
        super()._initialize(df, hat_H)

    def inference(
        self, 
        df: np.ndarray, 
        iterations: int = 3, 
        initialized: bool = False
    ) -> None:
        """Perform inference on df.
        
        Update parameters using Adam optimizer on penalized negative 
        log-likelihood. 

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
        def neg_log_like(pars, df):
            model = self._specify_filter(pars)
            return -jnp.mean(model.forward_filter(df)[2]) 
        
        if not initialized:
            self.initialize(df)
        pars = (self._log_rate, self._k_p, self._theta_p, self._log_sd, 
                self._transformed_corr, self._log_obs_sd)
        pars = super()._inference(pars, df, neg_log_like, iterations)
        (self._log_rate, self._k_p, self._theta_p, self._log_sd, 
         self._transformed_corr, self._log_obs_sd) = pars


class AFINS(AFNS):
    """Arbitrage-free independent Nelson-Siegel yield model.
    
    This is the independent 3-factor model from Christensen et al. (2011).
    """
    def __init__(
        self, 
        maturities: List[float],
        delta_t: float = 1/250,
        decay_rate: float = 0.8
    ) -> None:
        """Instantiate the class.
        
        Parameters
        ----------
        maturities : List[float]
            All Time-to-maturity of interest.
        delta_t : float, optional
            Time span between two observations. The default is 1/250.
        decay_rates : float
            The decay rate.

        Returns
        -------
        None
        """
        super().__init__(
            maturities=maturities,
            delta_t=delta_t,
            decay_rate=decay_rate
        )
        self._k_p_diag = jnp.diag(self._k_p)
        
    def specify_adjustments(self) -> float:
        """Specify the yield adjustment term of AFNS.

        Parameters
        ----------
        cov_mat: np.ndarray
            Continuous-time covariance matrix, shape = [dim_x, dim_x].
        
        Returns
        -------
        float
            Yield adjustment terms.
        """
        cov_mat = super()._independent_continuous_covariance(self._log_sd)
        return self._specify_adjustments(cov_mat, jnp.exp(self._log_rate))
    
    def specify_filter(self) -> kf.BaseLGSSM:
        """Specify the LGSSM given the parameter values. 

        Returns
        -------
        kf.BaseLGSSM
            The LGSSM. 
        """
        return self._specify_filter((
            self._log_rate, self._k_p_diag, self._theta_p, self._log_sd, 
            self._log_obs_sd
        ))
        
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
        log_rate, k_p_diag, theta_p, log_sd, log_obs_sd = pars
        k_p = jnp.diag(k_p_diag)
        cov_mat = super()._independent_continuous_covariance(log_sd)
        (hat_A, hat_F, hat_Q), hat_R, (hat_m0, hat_P0) = super()._discrete_components(
            cov_mat, (k_p, theta_p, log_obs_sd)
        )
        
        hat_B, hat_H = self._observation_components(cov_mat, jnp.exp(log_rate))
        
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
        super().initialize(df)
        self._k_p_diag = jnp.diag(self._k_p)
        
    def inference(
        self, 
        df: np.ndarray, 
        iterations: int = 3, 
        initialized: bool = False
    ) -> None:
        """Perform inference on df.
        
        Update parameters using Adam optimizer on penalized negative 
        log-likelihood. 

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
        def neg_log_like(pars, df):
            model = self._specify_filter(pars)
            return -jnp.mean(model.forward_filter(df)[2]) 
        
        if not initialized:
            self.initialize(df)
        pars = (self._log_rate, self._k_p_diag, self._theta_p, self._log_sd, 
                self._log_obs_sd)
        pars = super()._inference(pars, df, neg_log_like, iterations)
        (self._log_rate, self._k_p_diag, self._theta_p, self._log_sd, 
         self._log_obs_sd) = pars