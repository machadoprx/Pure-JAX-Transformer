import jax
import jax.numpy as jnp
import numpy as np
from collections import abc

def optimizer_sgd_tr(params, grads, skip_upd, lr):
    for key, value in params.items():
        if isinstance(value, abc.Mapping):
            optimizer_sgd_tr(value, grads[key], skip_upd, lr)
        elif key in skip_upd:
            return
        else:
            params[key] = params[key] - (lr * jnp.mean(grads[key], axis=0))
    return params