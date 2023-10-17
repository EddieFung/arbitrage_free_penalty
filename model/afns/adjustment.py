from typing import Tuple

import jax
import jax.numpy as jnp


@jax.jit
def adjustment_1(m: float, rate: float) -> float:
    return m / 4 / rate + jnp.exp(-m * rate) / 2 / rate**2 - \
        (1 - jnp.exp(-m * rate)) / 2 / rate**3 / m

@jax.jit
def adjustment_2(m: float, rate: float):
    return 3 * jnp.exp(-m * rate) / 2 / rate**2 + m / 4 / rate + \
        m * jnp.exp(-m * rate) / 2 / rate - \
            3 * (1 - jnp.exp(-m * rate)) / 2 / rate**3 / m

@jax.jit
def adjustment_3(m: float, rates: Tuple[float, float]) -> float:
    pass

@jax.jit
def adjustment_4(m: float, rates: Tuple[float, float]) -> float:
    pass

@jax.jit
def adjustment_5(m: float, rates: Tuple[float, float]) -> float:
    pass