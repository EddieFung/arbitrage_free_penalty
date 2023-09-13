from typing import List, Sequence, Callable

import numpy as np

import jax
import jax.numpy as jnp
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
        self.initialize(df)
        
        sobolev_weights_m = jnp.log(jnp.diff(self.maturities, prepend=0)) - \
            self.maturities
        
        samples = sampling_method(no_samples)
        sobolev_weights_b = - jnp.sum(samples**2, 1)**0.5 - lpdf(samples) - \
            jnp.log(no_samples)
        
    
    
    