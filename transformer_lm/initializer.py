import jax
import jax.numpy as jnp
import numpy as np

def get_linear_params(rng, in_feat, out_feat, bias=False):
	rng, subkey = jax.random.split(rng)
	init = jax.nn.initializers.glorot_normal()
	weights = init(subkey, (in_feat, out_feat), jnp.float32)
	biases = None
	if bias:
		biases = jnp.zeros((1, out_feat))
	return rng, weights, biases

def get_ff_block_params(rng, in_feat, out_feat):
	rng, w1, b1 = get_linear_params(rng, in_feat, out_feat, bias=True)
	rng, w2, b2 = get_linear_params(rng, out_feat, out_feat, bias=True)
	rng, _ = jax.random.split(rng)
	return rng, {'W1':w1, 'W2':w2, 'b1':b1, 'b2':b2}

def get_mha_params(rng, dk, dv, out_features, num_heads):
	rng, WQs, _ = get_linear_params(rng, out_features, num_heads * dk, bias=False)
	rng, WKs, _ = get_linear_params(rng, out_features, num_heads * dk, bias=False)
	rng, WVs, _ = get_linear_params(rng, out_features, num_heads * dv, bias=False)
	rng, Wout, _ = get_linear_params(rng, num_heads * dv, out_features, bias=False)
	rng, _ = jax.random.split(rng)
	return rng, {'WQs':WQs, 'WKs':WKs, 'WVs':WVs, 'Wout':Wout, 'num_heads':num_heads}

def get_ln_params(seq_len):
	gamma = jnp.ones((seq_len, 1))
	beta = jnp.zeros((seq_len, 1))
	return {'gamma':gamma, 'beta':beta}

def get_transformer_params(rng, seq_len, dk, dv, hid_size, num_heads, num_layers, vocab_size, ff_out):
	params = {'num_layers':num_layers}
	rng, subkey = jax.random.split(rng)
	init = jax.nn.initializers.glorot_normal()

	params['embed'] = {}
	params['embed']['W'] = init(subkey, (hid_size, vocab_size), jnp.float32)

	for i in range(num_layers):
		rng, params_mha_enc = get_mha_params(rng, dk, dv, hid_size, num_heads)
		rng, params_ff_block_enc = get_ff_block_params(rng, hid_size, ff_out)
		params_ln_enc_1 = get_ln_params(seq_len)
		params_ln_enc_2 = get_ln_params(seq_len)
		params[f'encoder_{i}_mha'] = params_mha_enc
		params[f'encoder_{i}_ff_block'] = params_ff_block_enc
		params[f'encoder_{i}_ln_1'] = params_ln_enc_1
		params[f'encoder_{i}_ln_2'] = params_ln_enc_2

		rng, params_mha_dec_1 = get_mha_params(rng, dk, dv, hid_size, num_heads)
		rng, params_mha_dec_2 = get_mha_params(rng, hid_size, hid_size, hid_size, num_heads)
		rng, params_ff_block_dec = get_ff_block_params(rng, hid_size, ff_out)
		params_ln_dec_1 = get_ln_params(seq_len)
		params_ln_dec_2 = get_ln_params(seq_len)
		params_ln_dec_3 = get_ln_params(seq_len)

		params[f'decoder_{i}_mha_1'] = params_mha_dec_1
		params[f'decoder_{i}_mha_2'] = params_mha_dec_2
		params[f'decoder_{i}_ff_block'] = params_ff_block_dec
		params[f'decoder_{i}_ln_1'] = params_ln_dec_1
		params[f'decoder_{i}_ln_2'] = params_ln_dec_2
		params[f'decoder_{i}_ln_3'] = params_ln_dec_3

	return params

def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):

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

def get_causal_mask(seq_len):
	mask_tmp = jnp.tril(jnp.ones((seq_len,seq_len)))
	mask_causal = mask_tmp.at[jnp.where(mask_tmp == 1)].set(0)
	mask_causal = mask_causal.at[jnp.where(mask_tmp == 0)].set(-1e9)
	return mask_causal
	

