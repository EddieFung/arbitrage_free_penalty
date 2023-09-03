import numpy as np
import jax.numpy as jnp


def ns_basis(decay_rate: float, maturity: float):
    if maturity == 0.:
        return np.array([1., 1., 0.])
    return np.array([
        1,
        (1. - jnp.exp(-decay_rate * maturity)) / decay_rate / maturity,
        (1. - jnp.exp(-decay_rate * maturity)) / decay_rate / maturity - \
            jnp.exp(-decay_rate * maturity)
    ])

def arbitrage_free_yield_adjustment(
        decay_rate: float, 
        sigma: float, 
        maturity: float
        ):
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
            