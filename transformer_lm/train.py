import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, vmap
from initializer import *
from tranformer_modules import *
from forward import *
from loss import *
from layers import *

def train_step(inputs, params, vocab_size):
	
	loss = lm_loss_fn(inputs, params, forward_transformer, vocab_size, training=True)
	return loss


num_heads = 2
seq_len = 4
dk = 20
dv = 20
hid_size = 20
vocab_size = 10

embed_size = 20
#in_feats = 128
bs = 2
rng = jax.random.PRNGKey(42)
np.random.seed(42)

params = get_transformer_params(rng, seq_len, dk, dv, hid_size, num_heads, 1, hid_size, rate_att=0.2, rate_ff=0.2, eps=1e-9)

rng = jax.random.PRNGKey(0)
init = jax.nn.initializers.glorot_normal()

params['embed'] = {}
params['embed']['W'] = init(rng, (embed_size, vocab_size), jnp.float32)
rng, subkey = jax.random.split(rng)
params['linear_out_weights'] = init(subkey, (hid_size, vocab_size), jnp.float32)
params['linear_out_bias'] = jnp.zeros((1, vocab_size))

#print(params.keys())
#quit()

Q = jnp.array([[i+1 for i in range(seq_len)], [i+1 for i in range(seq_len)]])
Q_dec = jnp.array([[0, 4, 3, 2], [0, 4, 3, 2]])
targets = jnp.array([[4, 3, 2, 1], [4, 3, 2, 1]])
mask_tmp = jnp.tril(jnp.ones((seq_len,seq_len)))

#loss = vmap(train_loop, in_axes=([0, 0, 0, None], None, None))([Q, Q_dec, targets, mask_tmp], params, vocab_size)
#print(loss)
loss, grads = vmap(jax.value_and_grad(train_step, 1, allow_int=True), in_axes=([0, 0, 0, None], None, None))([Q, Q_dec, targets, mask_tmp], params, vocab_size)
print(jnp.mean(grads['linear_out_weights'], axis=0).shape)
