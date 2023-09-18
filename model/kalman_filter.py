from typing import Tuple, List, Callable
import numpy as np

import jax
import jax.numpy as jnp
import jax.scipy as jsc
from jax.lax import scan
from jax.example_libraries import optimizers


class BaseLGSSM:
    """Base class of Linear Gaussian State-space model.

    x_{t+1} = A + F x_t + noises, Var(noises) = Q,
    y_t = B + H x_t + noises, Var(noises) = R,
    x_0 ~ N(m0, P0)
    """
    def __init__(
        self, 
        A: np.array, 
        F: np.ndarray,
        Q: np.ndarray,
        B: np.array, 
        H: np.ndarray, 
        R: np.ndarray, 
        m0: np.array, 
        P0: np.ndarray
    ) -> None:
        """Instantiate the class.

        Parameters
        ----------
        A : np.array
            Transition intercept, shape = [dim_x].
        F : np.ndarray
            Transition matrix, shape = [dim_x, dim_x].
        Q : np.ndarray
            Transition covariance matrix, shape = [dim_x, dim_x].
        B : np.array
            Observation intercept, shape = [dim_y].
        H : np.ndarray
            Observation matrix, shape = [dim_y, dim_x].
        R : np.ndarray
            Observation covariance matrix, shape = [dim_y, dim_y].
        m0 : np.array
            Mean of initial distribution, shape = [dim_x].
        P0 : np.ndarray
            Covariance matrix of initial distribution, shape = [dim_x, dim_x].

        Returns
        -------
        None.
        """
        self.A = A
        self.F = F
        self.Q = Q
        self.B = B
        self.H = H
        self.R = R
        self.m0 = m0
        self.P0 = P0

        self.dim_x = self.Q.shape[0]
        self.dim_y = self.R.shape[0]

    def one_step_filter(
        self, 
        carry: Tuple[np.array, np.ndarray, float], 
        observation: np.array
    ) -> Tuple:
        """Perform Kalman filter for one time step.
        
        Estimate the condition density of x_t:
        p(x_t|x_{t-1},y_t) \propto p(y_t|x_t)p(x_t|x_{t-1})p(x_{t-1}|y_{1:t-1})
        
        Parameters
        ----------
        carry : Tuple[np.array, np.ndarray, float]
            Mean, covariance matrix, log-likelihood of p(x_{t-1}|y_{1:t-1})
        observation : np.array
            Observation at time t, shape = [dim_y]

        Returns
        -------
        Tuple
            Mean, covariance matrix, log-likelihood of p(x_{t}|y_{1:t}).
        """
        m, P, likeli = carry
        m = self.A + self.F @ m
        P = self.F @ P @ self.F.T + self.Q

        residuals = observation - self.B - self.H @ m
        S = self.H @ P @ self.H.T + self.R

        # Evaluate unnormalized log-likelihood
        lower = jsc.linalg.cholesky(S).T
        log_det = 2 * jnp.sum(jnp.log(jnp.abs(jnp.diag(lower))))
        scaled_diff = jsc.linalg.solve_triangular(lower, residuals, lower=True)
        distance = jnp.sum(scaled_diff * scaled_diff, 0)
        likeli = -0.5 * (distance + log_det)

        kalman_gain = jsc.linalg.solve(S, self.H @ P, assume_a='pos').T
        m = m + kalman_gain @ (observation - self.B - self.H @ m)
        P = P - kalman_gain @ S @ kalman_gain.T
        return (m, P, likeli), (m, P, likeli)

    def forward_filter(
        self, df: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.array]:
        """Perform Kalman filter on df.

        Parameters
        ----------
        df : np.ndarray
            Data, shape = [dim_t, dim_y].

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.array]
            Fitered mean, shape = [dim_t, dim_x];
            filtered covariance matrix, shape = [dim_t, dim_x, dim_x];
            normalized log-likelihood, shape = [dim_t]

        """
        _, (fms, fPs, likeli) = scan(
            self.one_step_filter, (self.m0, self.P0, 0.), df
        )
        return fms, fPs, likeli - self.dim_y / 2 * jnp.log(2 * jnp.pi)

    
class OUTransitionModel:
    """LGSSM with the discretized OU process as the transition process.

    Only the transition equation is fixed as the independent OU process. The 
    observation matrix and intercept are unspecified. This is a base class for
    other models.
    
    x_{t+1} = theta*(I-e^{-K delta}) + e^{-K delta} x_t + noises, 
    Var(noises) = \int^delta_0 e^{-K s} Sigma Sigma' e^{-K s} ds
    """
    def __init__(
        self, delta_t: float = 1/250
    ) -> None:
        """Instantiate the class.

        Parameters
        ----------
        delta_t : float, optional
            Time span between two observations. The default is 1/250.

        Returns
        -------
        None
        """
        self.delta_t = delta_t
        
        self._log_k = np.array([2., 2., 2.])
        self._theta = np.array([0., 2., 1.])
        self._log_sigma = np.array([-0.5, -0.5, -0.5])
        self._log_obs_sd = None
        
    def get_transition_covariance_matrix(self) -> np.ndarray:
        """Get transition covariance matrix.
        
        Returns
        -------
        np.ndarray
            Transition covariance matrix.
        """
        k = jnp.exp(self._log_k)
        transition_mat_diag = jnp.exp(-k * self.delta_t)
        return jnp.diag(jnp.exp(2 * self._log_sigma) * \
                        (1 - transition_mat_diag**2) / (2*k))

    def _specify_transition(self, pars: List) -> Tuple:
        """Hidden method to specify the components of the LGSSM.
        
        Parameters
        ----------
        pars : List
            Parameter values.

        Returns
        -------
        Tuple
            Transition intercept, matrix, covariance matrix;
            Observation covariance matrix;
            initial distribution mean, covariance matrix.
        """
        log_k, theta, log_sigma, log_obs_sd = pars
        transition_mat_diag = jnp.exp(-jnp.exp(log_k) * self.delta_t)
        hat_A = theta * (1 - transition_mat_diag)
        hat_F = jnp.diag(transition_mat_diag)
        hat_Q = jnp.diag(jnp.exp(2 * log_sigma - log_k) * \
                         (1 - transition_mat_diag**2) / 2)
        hat_R = jnp.diag(jnp.exp(2 * log_obs_sd))
        hat_m0 = theta
        hat_P0 = jnp.diag(jnp.exp(2 * log_sigma - log_k) / 2)
        
        return (hat_A, hat_F, hat_Q), hat_R, (hat_m0, hat_P0)
    
    def _initialize(self, df: np.ndarray, H: np.ndarray) -> None:
        """Initialize parameters given df.

        Parameters
        ----------
        df : np.ndarray
            Data, shape = [dim_t, dim_y].
        H : np.ndarray
            Observation matrix, shape = [dim_y, dim_x].

        Returns
        -------
        None
        """
        # initialized latent states
        states = jnp.linalg.lstsq(H, df.T)[0].T  # shape = [dim_t, dim_x]
  
        # initialized k while ensuring stationarity
        non_stationary_transition_mat_diag = jnp.diag(
            jnp.linalg.lstsq(states[:-1,], states[1:,])[0]
        )  # shape = [dim_x]
        transition_mat_diag = np.maximum(
            np.minimum(non_stationary_transition_mat_diag, 0.99), 0.01
        )
        k = - jnp.log(transition_mat_diag) / self.delta_t  # shape = [3]
        self._log_k = jnp.log(k)
  
        # initialized sigma
        transition_var = jnp.var(states[1:,] - \
                                 states[:-1,] * transition_mat_diag, 0)
        self._log_sigma = 0.5 * jnp.log(
            transition_var / (1. - transition_mat_diag**2) * 2. * k
        )  # shape = [3]
  
        self._theta = jnp.mean(states, 0)  # shape = [dim_x]
  
        # two initializers on the observation std:
        # 1. the sample std, which is an upper bound
        # 2. the residual std using the initialized states
        obs_sd_est_1 = jnp.log(jnp.std(df, 0))
        obs_sd_est_2 = jnp.log(jnp.std(states @ H.T - df, 0))
        self._log_obs_sd = (obs_sd_est_1 + obs_sd_est_2) / 2.
        
    def _inference(
            self, 
            pars: List, 
            df: np.ndarray, 
            loss: Callable, 
            iterations: int = 3
    ) -> Tuple:
        """Perform inference on df.
        
        Update parameters using Adam optimizer on loss (-ve log-likelihood).

        Parameters
        ----------
        pars : List
            Trainable parameters.
        df : np.ndarray
            Data, shape = [dim_t, dim_y].
        loss : Callable
            Negative log-likelihood of the model.
        iterations : int, optional
            Number of iterations of Adam optimizer. The default is 3.

        Returns
        -------
        Tuple
            Parameter values.
        """
        opt_init, opt_update, get_params = optimizers.adam(1e-3)
        opt_state = opt_init(pars)

        @jax.jit
        def step(step_idx, _opt_state, df):
            value, grads = jax.value_and_grad(loss)(
                get_params(_opt_state), df
            )
            _opt_state = opt_update(step_idx, grads, _opt_state)
            return _opt_state 

        for i in range(iterations):
            opt_state = step(i, opt_state, df)

        return get_params(opt_state)
    
class OUModel(OUTransitionModel):
    """LGSSM with the OU process, fixed observation matrix and intercept.
    """
    def __init__(
        self, 
        B: np.array, 
        H: np.ndarray, 
        delta_t: float = 1/250
    ) -> None:
        """Instantiate the class.

        Parameters
        ----------
        B : np.array
            Observation intercept, shape = [dim_y].
        H : np.ndarray
            Observation matrix, shape = [dim_y, dim_x].
        delta_t : float, optional
            Time span between two observations. The default is 1/250.

        Returns
        -------
        None
        """
        super().__init__(delta_t)
        self.B = B
        self.H = H
        self._log_obs_sd = np.zeros(len(B))
        
    def specify_filter(self) -> BaseLGSSM:
        """Specify the LGSSM given the parameter values. 

        Returns
        -------
        BaseLGSSM
            The LGSSM. 
        """
        return self._specify_filter([
            self._log_k, self._theta, self._log_sigma, self._log_obs_sd
        ])
        
    def _specify_filter(self, pars: List) -> BaseLGSSM:
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
        (hat_A, hat_F, hat_Q), hat_R, (hat_m0, hat_P0) = super()._specify_transition(pars)
        return BaseLGSSM(hat_A, hat_F, hat_Q, self.B, self.H, hat_R, hat_m0, hat_P0)
    
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
        super()._initialize(df, self.H)

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
            Data, shape = [dim_t, dim_y].
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
        pars = [self._log_k, self._theta, self._log_sigma, self._log_obs_sd]
        pars = super()._inference(pars, df, neg_log_like, iterations)
        self._log_k, self._theta, self._log_sigma, self._log_obs_sd = pars
