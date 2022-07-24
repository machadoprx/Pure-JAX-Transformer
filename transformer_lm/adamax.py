import jax
import jax.numpy as jnp

def adamax(params, grads, state, step, lr=1e-4, delta1=0.9, delta2=0.999):

    eps = 1e-7
    '''params_v = {
        'wv': np.zeros(X.shape[1:]),
        'bv': 0.
    }

    # Instead of sum of squares (similar to l2 norm), adamax uses l-infinity norm
    infinity_grad = {
        'w': np.zeros(X.shape[1:]),
        'b': 0.
    }
    # The leaves in value_flat correspond to the `*` markers in value_tree
    value_flat, value_tree = tree_flatten(value_structured)
    print("value_flat={}\nvalue_tree={}".format(value_flat, value_tree))

    # Transform the flat value list using an element-wise numeric transformer
    transformed_flat = list(map(lambda v: v * 2., value_flat))
    print("transformed_flat={}".format(transformed_flat))

    # Reconstruct the structured output, using the original
    transformed_structured = tree_unflatten(value_tree, transformed_flat)
    print("transformed_structured={}".format(transformed_structured))
    '''
    # params = jax.tree_util.tree_map(lambda p, g: p - lr * jnp.mean(g, axis=0) if not isinstance(p, int) else p, params, grads)
    # Momements update

    params_flat, tree = jax.tree_util.tree_flatten(params)
    grads_flat, _ = jax.tree_util.tree_flatten(grads)
    state_mom_flat, _ = jax.tree_util.tree_flatten(state['mom'])
    state_inf_flat, _ = jax.tree_util.tree_flatten(state['inf'])
    
    for i, param, grad, state_mom, state_inf in enumerate(zip(params_flat, grads_flat, state_mom_flat, state_inf_flat), 0):

        state_mom_flat[i] = delta1 * state_mom + (1. - delta1) * jnp.mean(grad, axis=0)

        state_inf_flat[i] = jnp.maximum(delta2 * state_inf, jnp.abs(grad_w))

        # Bias correction
        moment_w = params_v['wv'] / (1. - delta1**(step+1))
        moment_b = params_v['bv'] / (1. - delta1**(step+1))


        params['w'] -= (lr / (jnp.sqrt(infinity_grad['w']) + eps)) * moment_w
        params['b'] -= (lr / (jnp.sqrt(infinity_grad['b']) + eps)) * moment_b

    return params