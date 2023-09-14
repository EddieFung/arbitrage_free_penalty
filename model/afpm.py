from typing import List, Sequence, Callable

import numpy as np

import jax
import jax.numpy as jnp
import jax.scipy as jsc
import flax.core
import flax.linen as nn
from jax import value_and_grad
from jax.example_libraries import optimizers

from model import utils
from model import kalman_filter as kf

class MLP(nn.Module):
    """
    Multilayer perceptron using sigmoid activation.
    
    :params features: number of neurons in each layer.
    """
    features: Sequence[int]

    @nn.compact
    def __call__(self, x: float):
        """Evaluate the model at x."""
        for feat in self.features[:-1]:
            x = nn.sigmoid(nn.Dense(feat)(x))
        x = nn.Dense(self.features[-1])(x)
        return x

class AFPM:
    def __init__(
        self,
        features: Sequence[int],
        reference_basis: Callable,
        maturities: List[float],
        delta_t: float = 1/250,
        start: float = 0,
        stop: float = 30,
        num: int = 500
    ):
        assert reference_basis[-1] == features - 1
        
        self.grids = utils.Grids(start, stop, num)
        self.nn = MLP(features)
        self.nn_pars = self.nn.init(jax.random.PRNGKey(42), jnp.ones(1)) 
        self.ou_model = kf.OUModel(
            B=np.zeros(len(maturities)),
            H=np.array([reference_basis(m) for m in maturities]),
            delta_t=delta_t
        )
    
    def initialize_ssm(self, df: np.ndarray):
        self.ou_model.initialize(df)
    
    def initialize_nn(self, fine_grid: int = 1000, iterations: int = 10000):
        maturities = np.linspace(
            np.max([self.start, 1/12]), np.min([self.start, 30]), fine_grid
        ).reshape((fine_grid,1))
        np.random.shuffle(maturities)
        
        target = np.array([self.reference_basis(np.squeeze(m)) 
                           for m in maturities])
        target = np.hstack([ # the int term is random
            np.random.uniform(low=-0.2,high=-0.001,size=(fine_grid,1)), target
        ])

        @jax.jit
        def nn_initial_loss(w):
            return jnp.mean((self.nn.apply(w, maturities) - target)**2)
        
        @jax.jit
        def initialization_fit(step, _opt_state):
            _, grads = value_and_grad(nn_initial_loss)(get_params(_opt_state))
            _opt_state = opt_update(step, grads, _opt_state)
            return _opt_state
        
        opt_init, opt_update, get_params = optimizers.adam(1e-3)
        opt_state = opt_init(self.nn.weights)
        for i in range(iterations):
            opt_state = initialization_fit(i, opt_state)
        
        self.nn.pars = get_params(opt_state)
        
    def initialize(self, df: np.ndarray):
        self.initialize_ssm(df)
        self.initialize_nn()
    
    def inference(
        self,
        df: np.ndarray,
        sampling_method: Callable[[int], np.ndarray],
        lpdf: Callable,
        no_samples: int,
        iterations: int,
        penalty: float,
        fixed_volatility: bool = True
    ):
        @jax.jit
        def loss(
            w: flax.core.variables,
            betas: np.ndarray,
            sobolev_weights_b: np.array,
            sobolev_weights_m: np.array,
            covariance_mat: np.ndarray,
            penalty: float
        ):
            # distance loss
            log_weights = sobolev_weights_b[np.newaxis, :] + \
                sobolev_weights_m[:, np.newaxis]  # shape = [m, beta_size]
            est_basis = self.nn.apply(w, self.maturities[:, np.newaxis])  # shape=[m, d+1]
            distance_loss = jnp.exp(jsc.special.logsumexp(
                log_weights + 2 * jnp.log(est_basis[:, [0]] + \
                jnp.matmul(self.reference_basis - est_basis[:, 1:], betas.T))
            ))
        
            # arbitrage-free loss
            integrated_basis = self.grids.integrate( 
                lambda x: self.nn.apply(w, x[:, np.newaxis])
            )[:, 1:] # intercept term is ignored, shape = [m, d]
            full_grids_basis = - self.nn.apply(w, np.zeros((1,1))) + \
                self.nn.apply(w, self.grids.grids[:, np.newaxis])
            grids_basis = full_grids_basis[:, 1:]  # shape = [m, d]
            grids_int_basis = full_grids_basis[:, [0]]  # shape = [m, 1]
            
            # risk-neutral drift, shape=[d, d]
            neutral_drift = jnp.linalg.lstsq(integrated_basis, grids_basis)[0]
        
            AF_loss_1 = (grids_int_basis + 0.5 * jnp.sum(
                jnp.matmul(integrated_basis, covariance_mat) * \
                    integrated_basis, 1, keepdims=True
            ))**2
            AF_loss_2 = jnp.sum(
                (grids_basis - jnp.matmul(integrated_basis, neutral_drift))**2,
                1, keepdims=True
            )
            
            AF_loss = AF_loss_1 + AF_loss_2
        
            integrated_AF_loss = jnp.exp(1 / penalty * jsc.special.logsumexp(
                jnp.log(self.grids.stepsize) - self.grids.grids + \
                    penalty * jnp.log(AF_loss)
            ))
            return distance_loss + integrated_AF_loss
        
        @jax.jit
        def fit(
            step: int,
            _opt_state, 
            betas: np.ndarray,
            sobolev_weights_b: np.array,
            sobolev_weights_m: np.array, 
            covariance_mat: np.ndarray,
            penalty: float
        ):
            value, grads = value_and_grad(loss)(
                w=get_params(_opt_state), 
                betas=betas, 
                sobolev_weights_b=sobolev_weights_b,
                sobolev_weights_m=sobolev_weights_m,
                covariance_mat=covariance_mat,
                penalty=penalty
            )
            _opt_state = opt_update(step, grads, _opt_state)
            return _opt_state
        
        self.initialize(df)
        opt_init, opt_update, get_params = optimizers.adam(1e-3)
        opt_state = opt_init(self.nn_pars)
        weights_m = jnp.log(jnp.diff(self.maturities, prepend=0)) - \
            self.maturities
            
        samples = sampling_method(no_samples)
        weights_b = - jnp.sum(samples**2, 1)**0.5 - lpdf(samples) - \
            jnp.log(no_samples)
        for i in range(iterations):
            opt_state = fit(
                step=i,
                _opt_state=opt_state, 
                betas=samples,
                sobolev_weights_b=weights_b,
                sobolev_weights_m=weights_m, 
                covariance_mat=self.ou_model.get_transiton_covariance_matrix(),
                penalty=penalty
            )

        self.nn_pars = get_params(opt_state)
