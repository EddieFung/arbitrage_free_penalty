import numpy as np

import jax
import jax.numpy as jnp


@jax.jit
def adjustment_matrix(m: float, rates: np.array) -> np.ndarray:
    if m == 0:
        return np.zeros([[5, 5]])
    return np.array([
        [m**2 / 6, fn_1(m, rates[0]), fn_1(m, rates[1]), 
         fn_2(m, rates[0]), fn_2(m, rates[1])],
        [fn_1(m, rates[0]), 
         fn_3(m, [rates[0], rates[0]]), fn_3(m, [rates[1], rates[0]]),
         fn_5(m, [rates[0], rates[0]]), fn_5(m, [rates[0], rates[1]])],
        [fn_1(m, rates[1]), 
         fn_3(m, [rates[0], rates[1]]), fn_3(m, [rates[1], rates[1]]),
         fn_5(m, [rates[1], rates[0]]), fn_5(m, [rates[1], rates[1]])],
        [fn_2(m, rates[0]), 
         fn_5(m, [rates[0], rates[0]]), fn_5(m, [rates[1], rates[0]]),
         fn_4(m, [rates[0], rates[0]]), fn_4(m, [rates[1], rates[0]])],
        [fn_2(m, rates[1]), 
         fn_5(m, [rates[0], rates[1]]), fn_5(m, [rates[1], rates[1]]),
         fn_4(m, [rates[0], rates[1]]), fn_4(m, [rates[1], rates[1]])]
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