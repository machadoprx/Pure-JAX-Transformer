import jax
import jax.numpy as jnp
from utils import *

def lm_loss_fn(inputs, params, forward_fn, vocab_size: int, training: bool = True) -> jnp.ndarray:
    
    x, targets = inputs
    #print(x.shape)
    #quit()
    #print(x, targets)
    #quit()
    mask_x = x == 0
    mask_target = targets == 0
    mask_ce = jnp.greater(targets != 0, 0)
    #print(mask_ce)

    logits = forward_fn([x, mask_x, targets, mask_target], params, training=training)

    targets = jax.nn.one_hot(targets, vocab_size)
    assert logits.shape == targets.shape

    loss = -jnp.sum(targets * jax.nn.log_softmax(logits), axis=-1)
    loss = jnp.sum(loss * mask_ce) / jnp.sum(mask_ce)

    return loss