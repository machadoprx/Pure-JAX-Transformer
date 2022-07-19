import jax
import jax.numpy as jnp
import numpy as np
from jax import grad
from initializer import *
from tranformer_modules import *
from forward import *
from loss import *
from layers import *

def train_loop(inputs, params, vocab_size):
	
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

Q = jnp.array([i+1 for i in range(seq_len)])
Q_dec = jnp.array([0, 4, 3, 2])
targets = jnp.array([4, 3, 2, 1])
mask_tmp = jnp.tril(jnp.ones((seq_len,seq_len)))

loss = train_loop([Q, Q_dec, targets, mask_tmp], params, vocab_size)
print(loss)
print(grad(train_loop, 1, allow_int=True)([Q, Q_dec, targets, mask_tmp], params, vocab_size))

'''mask_tmp = jnp.tril(jnp.ones((seq_len,seq_len)))
mask_causal = mask_tmp.at[jnp.where(mask_tmp == 1)].set(0)
mask_causal = mask_causal.at[jnp.where(mask_tmp == 0)].set(-1e9)
#print(mask_causal)
rng, emb_params, _ = get_linear_params(rng, 20, 4, bias=False)
Q = jnp.array([i+1 for i in range(4)])
pos_enc = get_sinusoid_encoding_table(seq_len, hid_size, padding_idx=None)
Q = embed([Q, pos_enc], {'W':emb_params})
print(pos_enc)
print(Q)
print(Q.shape)
#quit()
mask = None

inputs = [Q, Q, Q, mask]

out = encoder_block(inputs, params['encoder'], training=True)
print(out)
print(out.shape)
Q_dec = jnp.array([i for i in range(4)])
Q_dec = embed([Q_dec, pos_enc], {'W':emb_params})

print(Q)
print(Q_dec)
inputs_decoder = [Q_dec, Q_dec, Q_dec, out, mask_causal]
out_dec = decoder_block(inputs_decoder, params['decoder'], training=True)
print(out_dec)
print(out_dec.shape)'''