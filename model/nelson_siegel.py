import numpy as np
import jax.numpy as jnp


def yield_basis(decay_rate: float, maturity: float):
    """The three Nelson-Siegel basis functions for Yields.

    Parameters
    ----------
    decay_rate : float
        lambda.
    maturity : float
        time-to-maturity.

    Returns
    -------
    np.array
        Three basis functions for yields.

    """
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
            
def forward_basis(decay_rate: float, maturity: float):
    """The three Nelson-Siegel basis functions for forward rates.

    Parameters
    ----------
    decay_rate : float
        lambda.
    maturity : float
        time-to-maturity.

    Returns
    -------
    np.array
        Three basis functions for forward rates.

    """
    return np.array([
        1,
        jnp.exp(-decay_rate * maturity),
        decay_rate * maturity * jnp.exp(-decay_rate * maturity)
    ])