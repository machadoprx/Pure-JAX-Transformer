import jax
import jax.numpy as jnp
import numpy as np
from initializer import *
from tranformer_modules import *
from layers import *

def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    '''  '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

   

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return jnp.array(sinusoid_table)

def test():
	num_heads = 2
	seq_len = 4
	dk = 20
	dv = 20
	hid_size = 20
	#in_feats = 128
	bs = 2
	rng = jax.random.PRNGKey(42)
	np.random.seed(42)

	params = get_transformer_params(rng, seq_len, dk, dv, hid_size, num_heads, 1, hid_size, rate_att=0.2, rate_ff=0.2, eps=1e-9)
	params = params[0]
	mask_tmp = jnp.tril(jnp.ones((seq_len,seq_len)))
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
	print(out_dec.shape)
	'''out, _ = multihead_attention(inputs, params_mha, training=True)
	print(out.shape)
	out = layer_norm(out + Q, params_ln, training=True)
	print(out.shape)
	out = ff_block(out, params_ff_block, training=True)
	print(out.shape)'''

test()