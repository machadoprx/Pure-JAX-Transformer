import jax
import jax.numpy as jnp

@jax.jit
def adamax(params, grads, state, step, lr=1e-4, delta1=0.9, delta2=0.999):

    eps = 1e-8
    params_flat, tree = jax.tree_util.tree_flatten(params)
    grads_flat, _ = jax.tree_util.tree_flatten(grads)
    state_mom_flat, _ = jax.tree_util.tree_flatten(state['mom'])
    state_inf_flat, _ = jax.tree_util.tree_flatten(state['inf'])
    
    for i, (param, grad, state_mom, state_inf) in enumerate(zip(params_flat, grads_flat, state_mom_flat, state_inf_flat), 0):
        grad = jnp.mean(grad, axis=0)
        state_mom = delta1 * state_mom + (1. - delta1) * grad
        state_inf = jnp.maximum(delta2 * state_inf, jnp.abs(grad))
        curr_lr = lr / (1. - delta1**(step+1))
        param = param - ((curr_lr * state_mom) / (state_inf + eps))

        params_flat[i] = param
        state_mom_flat[i] = state_mom
        state_inf_flat[i] = state_inf

    state['mom'] = jax.tree_util.tree_unflatten(tree, state_mom_flat)
    state['inf'] = jax.tree_util.tree_unflatten(tree, state_inf_flat)
    params = jax.tree_util.tree_unflatten(tree, params_flat)

    return params, state

def lr_schedule(d_model, step, warmup_steps=400):
    x1 = 1/jnp.sqrt(step)
    x2 = step * (warmup_steps ** -1.5)
    if x1 > x2:
        out = x2
    else:
        out = x1
    return 1/jnp.sqrt(d_model) * out