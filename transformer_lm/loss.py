import jax
import jax.numpy as jnp

def lm_loss_fn(inputs, params, forward_fn, vocab_size: int, training: bool = True) -> jnp.ndarray:
    
    inputs_enc, inputs_dec, targets, mask = inputs
    logits = forward_fn([inputs_enc, inputs_dec], params, training=training)
    targets = jax.nn.one_hot(targets, vocab_size)
    assert logits.shape == targets.shape

    mask = jnp.greater(mask, 0) # ?
    loss = -jnp.sum(targets * jax.nn.log_softmax(logits), axis=-1)
    loss = jnp.sum(loss * mask) / jnp.sum(mask)

    return loss