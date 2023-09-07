import numpy as np
import jax.numpy as jnp

from model import nelson_siegel as ns
from model import afns

class DataSimulator:
    
    def __init__(self):
        self.T = 500
        self.delta_t = 1 / 250
        self.maturities = np.array([1/12, 1/6, 1/4, 1/2, 1, 2, 3, 5, 7, 10, 20, 30])
        
        self.decay_rate = 0.5975
        self.k = np.array([0.0816, 0.2114, 1.2330])
        self.theta = np.array([0.0710, -0.0282, -0.0093])
        self.sigma = np.array([0.0051, 0.0110, 0.0264])
        
        self.transition_matrix = np.diag(jnp.exp(-self.k * self.delta_t))
        self.transition_int = (1. - jnp.exp(-self.k * self.delta_t)) * self.theta
        self.transition_std = np.diag(self.sigma * \
            ((1 - jnp.exp(-self.k * self.delta_t * 2)) / (2 * self.k))**0.5
        )
            
        self.observation_matrix = np.array([
            ns.yield_basis(self.decay_rate, m) for m in self.maturities
        ])
        self.observation_int = np.array([
            afns.arbitrage_free_yield_adjustment(self.decay_rate, self.sigma, m)
            for m in self.maturities
        ])
        self.observation_std = np.diag(jnp.exp(np.random.uniform(
            low=-7, high=-5, size=len(self.maturities)
        )))
        
        self.initial_mean = self.theta - 1.3 * np.diag(self.transition_std)
        self.initial_std = self.transition_std * 1.4
        
        self.x = None
        self.df = None
        self.simulate()

    def simulate(self):
        x = new_x = np.random.multivariate_normal(
            self.initial_mean, self.initial_std**2
        )
        y = self.observation_int + self.observation_matrix @ new_x + \
            np.random.multivariate_normal(
                np.zeros(len(self.maturities)), self.observation_std**2
            )
        
        for t in range(self.T-1):
            new_x = self.transition_int + self.transition_matrix @ new_x + \
                np.random.multivariate_normal(np.zeros(3), self.transition_std**2)
            new_y = self.observation_int + self.observation_matrix @ new_x + \
                np.random.multivariate_normal(
                    np.zeros(len(self.maturities)), self.observation_std**2
                )
            
            x = np.vstack([x, new_x])
            y = np.vstack([y, new_y])
        
        self.x = x
        self.df = y