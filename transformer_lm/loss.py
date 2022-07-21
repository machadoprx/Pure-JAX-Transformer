import jax
import jax.numpy as jnp
from utils import *

def lm_loss_fn(inputs, params, forward_fn, vocab_size: int, training: bool = True) -> jnp.ndarray:
    
    inputs, targets = inputs
    seq_len = len(inputs)

    # causal mask only in mha; 0 as padding token id
    mask_input = inputs == 0
    mask_input = jnp.where(mask_input, -jnp.inf, jnp.zeros((seq_len,seq_len)))

    mask_target = targets == 0
    mask_target = jnp.where(mask_target, -jnp.inf, jnp.zeros((seq_len,seq_len)))
    
    logits = forward_fn([inputs, mask_input, targets, mask_target], params, training=training)

    mask_ce = targets == 0
    targets = jax.nn.one_hot(targets, vocab_size)
    assert logits.shape == targets.shape
    mask = jnp.greater(mask_ce, 0) # ?
    loss = -jnp.sum(targets * jax.nn.log_softmax(logits), axis=-1)
    loss = jnp.sum(loss * mask) / jnp.sum(mask)

    return loss