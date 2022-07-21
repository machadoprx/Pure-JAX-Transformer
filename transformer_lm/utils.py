import jax
import jax.numpy as jnp

'''def masked_fill(mask, a, fill):
	return jax.lax.select(mask, a, jax.lax.broadcast(fill, a.shape))

# pad token == 0
def get_mask(src, causal=False):
	seq_len = len(src)
	pad_mask = src

	mask_tmp = jnp.tril(jnp.ones((seq_len,seq_len)))
	mask_causal = mask_tmp.at[jnp.where(mask_tmp == 1)].set(0)
	mask_causal = mask_causal.at[jnp.where(mask_tmp == 0)].set(-1e9)
	return mask_causal

import numpy as np

inputs = np.array([15,5,0,0])
mask = inputs == 0
#mask_tmp = np.tril(inputs != 0, dtype=np.int32)
#mask_tmp[:, -2] = 0
print(mask)
print(inputs)
print(jnp.where(mask, -jnp.inf, jnp.zeros((4,4))))'''