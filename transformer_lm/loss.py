import jax
import jax.numpy as jnp

def mlm_loss_fn(inputs, params, hyper_params, forward_fn, vocab_size: int) -> jnp.ndarray:
    
    x, targets = inputs
    mask_x = x == 0

    mask_target = x == 2

    logits = forward_fn([x, mask_x], params, hyper_params)
    
    targets = jax.nn.one_hot(targets, vocab_size)
    
    assert logits.shape == targets.shape

    loss = -jnp.sum(targets * jax.nn.log_softmax(logits), axis=-1)
    loss = jnp.sum(loss * mask_target) / jnp.sum(mask_target)

    return loss

def lm_loss_fn(inputs, params, hyper_params, forward_fn, vocab_size: int) -> jnp.ndarray:
    
    x, targets = inputs
    mask_x = x == 0
    mask_target = targets == 0

    logits = forward_fn([x, mask_x, targets, mask_target], params, hyper_params)
    
    targets = targets[1:]
    logits = logits[:-1, :]
    mask_ce = jnp.greater(targets != 0, 0)
    targets = jax.nn.one_hot(targets, vocab_size)
    
    assert logits.shape == targets.shape

    loss = -jnp.sum(targets * jax.nn.log_softmax(logits), axis=-1)
    loss = jnp.sum(loss * mask_ce) / jnp.sum(mask_ce)

    return loss