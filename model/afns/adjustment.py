import numpy as np

import jax
import jax.numpy as jnp


def adjustment_matrix(m: float, rates: np.array) -> np.ndarray:
    """Matrix for yield adjustment term.
    
    In the 5-factor AFGNS model, the yield adjustment term (intercept) equals:
    -tr[Covariance matrix @ -this matrix]

    Parameters
    ----------
    m : float
        Time-to-maturity.
    rates : np.array
        The two decay rates.

    Returns
    -------
    np.ndarray
        adjustment matrix, shape = [5, 5].
    """
    if m == 0:
        return jnp.zeros([5, 5])
    
    rates_2 = jnp.repeat(rates[0], 2)
    rates_3 = jnp.repeat(rates[1], 2)
    rates_4 = jnp.flip(rates)
    return jnp.array([
        [m**2 / 6, fn_1(m, rates[0]), fn_1(m, rates[1]), 
         fn_2(m, rates[0]), fn_2(m, rates[1])],
        [fn_1(m, rates[0]), 
         fn_3(m, rates_2), fn_3(m, rates_4),
         fn_5(m, rates_2), fn_5(m, rates)],
        [fn_1(m, rates[1]), 
         fn_3(m, rates), fn_3(m, rates_3),
         fn_5(m, rates_4), fn_5(m, rates_3)],
        [fn_2(m, rates[0]), 
         fn_5(m, rates_2), fn_5(m, rates_4),
         fn_4(m, rates_2), fn_4(m, rates_4)],
        [fn_2(m, rates[1]), 
         fn_5(m, rates), fn_5(m, rates_3),
         fn_4(m, rates), fn_4(m, rates_3)]
    ])

def smaller_adjustment_matrix(m: float, rate: float) -> np.ndarray:
    """Matrix for yield adjustment term.
    
    In the 3-factor AFNS model, the yield adjustment term (intercept) equals:
    -tr[Covariance matrix @ -this matrix]

    Parameters
    ----------
    m : float
        Time-to-maturity.
    rate : float
        The decay rates.

    Returns
    -------
    np.ndarray
        adjustment matrix, shape = [3, 3].
    """
    if m == 0:
        return jnp.zeros([3, 3])
    
    rates = jnp.repeat(rate, 2)
    return jnp.array([
        [m**2 / 6, fn_1(m, rate), fn_2(m, rate)],
        [fn_1(m, rate), fn_3(m, rates), fn_5(m, rates)],
        [fn_2(m, rate), fn_5(m, rates), fn_4(m, rates)]
    ])

@jax.jit
def fn_1(m: float, rate: float) -> float:
    return m / 4 / rate + jnp.exp(-m * rate) / 2 / rate**2 - \
        (1 - jnp.exp(-m * rate)) / 2 / rate**3 / m

@jax.jit
def fn_2(m: float, rate: float) -> float:
    return 3 * jnp.exp(-m * rate) / 2 / rate**2 + m / 4 / rate + \
        m * jnp.exp(-m * rate) / 2 / rate - \
        3 * (1 - jnp.exp(-m * rate)) / 2 / rate**3 / m

@jax.jit
def fn_3(m: float, rates: np.array) -> float:
    neg_exp_m_rate = jnp.exp(- m * rates)
    prod_rates = jnp.prod(rates)
    return 1 / 2 / prod_rates - \
        (1 - neg_exp_m_rate[0]) / (2 * prod_rates * m * rates[0]) - \
        (1 - neg_exp_m_rate[1]) / (2 * prod_rates * m * rates[1]) + \
        (1 - jnp.prod(neg_exp_m_rate)) / (2 * prod_rates * m * jnp.sum(rates))

@jax.jit
def fn_4(m: float, rates: np.array) -> float:
    neg_exp_m_rate = jnp.exp(- m * rates)
    prod_rates = jnp.prod(rates)
    sum_rates = jnp.sum(rates)
    return (1 + jnp.sum(neg_exp_m_rate) - jnp.prod(neg_exp_m_rate)) / \
        (2 * prod_rates) - \
        jnp.prod(neg_exp_m_rate) / sum_rates**2 - \
        jnp.prod(neg_exp_m_rate) * m / 2 / sum_rates - \
        (1 - neg_exp_m_rate[0]) / (prod_rates * m * rates[0]) - \
        (1 - neg_exp_m_rate[1]) / (prod_rates * m * rates[1]) + \
        (1 - jnp.prod(neg_exp_m_rate)) / sum_rates**3 / m + \
        (1 - jnp.prod(neg_exp_m_rate)) / (prod_rates * sum_rates * m)  

@jax.jit
def fn_5(m: float, rates: np.array) -> float:
    neg_exp_m_rate = jnp.exp(- m * rates)
    prod_rates = jnp.prod(rates)
    sum_rates = jnp.sum(rates)
    return (1 + neg_exp_m_rate[1]) / 2 / prod_rates - \
        jnp.prod(neg_exp_m_rate) / (2 * rates[0] * sum_rates) - \
        (1 - neg_exp_m_rate[0]) / (2 * prod_rates * m * rates[0]) - \
        (1 - neg_exp_m_rate[1]) / (prod_rates * m * rates[1]) + \
        (1 - jnp.prod(neg_exp_m_rate)) * (sum_rates + rates[1]) / \
        (2 * prod_rates * sum_rates**2 * m)