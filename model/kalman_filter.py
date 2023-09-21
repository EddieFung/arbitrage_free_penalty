from typing import Tuple, Callable
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
    other models. Also, the off-diagonal correlation in Sigma is assumed 
    uniform and positive.
    
    x_{t+1} = theta*(I-e^{-K delta}) + e^{-K delta} x_t + noises, 
    Var(noises) = \int^delta_0 e^{-K s} Sigma Sigma' e^{-K s} ds
    """
    def __init__(
        self, dim_x: int = 5, delta_t: float = 1/250
    ) -> None:
        """Instantiate the class.

        Parameters
        ----------
        dim_x: int, optional
            Dimension of the latent factors. The default is 5.
        delta_t : float, optional
            Time span between two observations. The default is 1/250.

        Returns
        -------
        None
        """
        self.dim_x = dim_x
        self.delta_t = delta_t
        
        self._k_p = np.eye(self.dim_x)
        self._theta_p = np.zeros(self.dim_x)
        self._log_diag = - np.ones(self.dim_x) / 2.
        self._off_diag = np.zeros(int(self.dim_x * (self.dim_x - 1) / 2))
        self._log_obs_sd = None
        
    def _sepcify_continuous_dynamic(self, pars: Tuple) -> Tuple:
        """Hidden method to specify components in continuous-time dynamic.

        Parameters
        ----------
        pars : Tuple
            Parameter values: (k_p, log_diag, off_diag).

        Returns
        -------
        Tuple
            Square-root of the continuous-time Sigma, shape = dim_x, dim_x];
            Eigenvalues of k_p, shape = [dim_x],
            Eigenvectors of k_p, shape = dim_x, dim_x].
        """
        k_p, log_diag, off_diag = pars 
        
        # Estimate the square-root matrix
        sqrt_mat = jnp.zeros([self.dim_x, self.dim_x])
        sqrt_mat = sqrt_mat.at[jnp.tril_indices(self.dim_x, -1)].set(off_diag)
        sqrt_mat += jnp.diag(jnp.exp(log_diag))
        
        # eigendecomposition of k_p
        eig_val, eig_vector = jax.jit(jnp.linalg.eig)(k_p)  
        
        return sqrt_mat, eig_val, eig_vector
    
    def _sepcify_discrete_dynamic(self, pars: Tuple) -> Tuple:
        """Hidden method to specify components in discrete-time dynamic.
        
        Transition equation, initialization equation, and the observation
        covariance matrix are specified. Observation intercept and matrix are 
        left unspecified.
        
        Parameters
        ----------
        pars : Tuple
            Parameter values: (k_p, theta_p, log_diag, off_diag, log_obs_sd).

        Returns
        -------
        Tuple
            Transition intercept, matrix, covariance matrix;
            observation covariance matrix;
            initial distribution mean, covariance matrix.
        """
        k_p, theta_p, log_diag, off_diag, log_obs_sd = pars
        sqrt_mat, eig_val, eig_vector = self._sepcify_continuous_dynamic(
            (k_p, log_diag, off_diag)
        )
        
        hat_F = jnp.matmul(
            jnp.matmul(eig_vector.T, jnp.diag(jnp.exp(- self.delta_t * eig_val))), 
            eig_vector
        )
        hat_A = theta_p * (jnp.eye(self.dim_x) - hat_F)
        hat_R = jnp.diag(jnp.exp(2 * log_obs_sd))
        hat_m0 = theta_p
        
        # shape = [dim_x, dim_x] 
        sum_eig_val = eig_val[:, jnp.newaxis] + eig_val[jnp.newaxis, :]
        cov_mat = jnp.matmul(sqrt_mat, sqrt_mat.T)
        hat_P0 = jnp.matmul(
            jnp.matmul(eig_vector.T, cov_mat), eig_vector
        ) / sum_eig_val
        hat_Q = hat_P0 * \
            (1 - jnp.exp(-self.delta_t * sum_eig_val))
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
        # initialize latent states
        states = jnp.linalg.lstsq(H, df.T)[0].T  # shape = [dim_t, dim_x]
  
        # initialize k while ensuring stationarity
        non_stationary_transition_mat_diag = jnp.diag(
            jnp.linalg.lstsq(states[:-1,], states[1:,])[0]
        )  # shape = [dim_x]
        transition_mat_diag = np.maximum(
            np.minimum(non_stationary_transition_mat_diag, 0.99), 0.01
        )
        k = - jnp.log(transition_mat_diag) / self.delta_t  # shape = [3]
        self._k_p = jnp.diag(k)
  
        # initialize sigma, assume diagonal cov
        transition_var = jnp.var(states[1:,] - \
                                 states[:-1,] * transition_mat_diag, 0)
        self._log_diag = 0.5 * jnp.log(
            transition_var / (1. - transition_mat_diag**2) * 2. * k
        )  # shape = [dim_x]
  
        self._theta_p = jnp.mean(states, 0)  # shape = [dim_x]
  
        # two initializers on the observation std:
        # 1. the sample std, which is an upper bound
        # 2. the residual std using the initialized states
        obs_sd_est_1 = jnp.log(jnp.std(df, 0))
        obs_sd_est_2 = jnp.log(jnp.std(states @ H.T - df, 0))
        self._log_obs_sd = (obs_sd_est_1 + obs_sd_est_2) / 2.
        
    def _inference(
        self, 
        pars: Tuple, 
        df: np.ndarray, 
        loss: Callable, 
        iterations: int = 3
    ) -> Tuple:
        """Perform inference on df.
        
        Update parameters using Adam optimizer on loss (-ve log-likelihood).

        Parameters
        ----------
        pars : Tuple
            Trainable parameters.=
        df : np.ndarray
            Data, shape = [dim_t, dim_y].
        loss : Callable
            Negative log-likelihood of the model.
        iterations : int, optional
            Number of iterations of Adam optimizer. The default is 3.

        Returns
        -------
        Tuple
            Updated parameter values.
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
        super().__init__(dim_x=H.shape[1], delta_t=delta_t)
        self.B = B
        self.H = H
        self._log_obs_sd = - np.ones(len(B))
        
    def specify_filter(self) -> BaseLGSSM:
        """Specify the LGSSM given the parameter values. 

        Returns
        -------
        BaseLGSSM
            The LGSSM. 
        """
        return self._specify_filter([
            self._k_p, self._theta_p, 
            self._log_diag, self._off_diag, self._log_obs_sd
        ])
        
    def _specify_filter(self, pars: Tuple) -> BaseLGSSM:
        """Hidden method to specify the LGSSM.
        
        Parameters
        ----------
        pars : Tuple
            Parameter values.

        Returns
        -------
        BaseLGSSM
            The LGSSM. 
        """
        (hat_A, hat_F, hat_Q), hat_R, (hat_m0, hat_P0) = super()._sepcify_discrete_dynamic(pars)
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
        pars = [self._log_k, self._theta, 
                self._log_sigma, self._log_corr, self._log_obs_sd]
        pars = super()._inference(pars, df, neg_log_like, iterations)
        (self._log_k, self._theta,
         self._log_sigma, self._log_corr, self._log_obs_sd) = pars
