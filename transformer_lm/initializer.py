import jax
import jax.numpy as jnp

def get_linear_params(rng, in_feat, out_feat, bias=False):
	rnd_range = 1 / in_feat ** 0.5
	rng, subkey = jax.random.split(rng)
	weights = jax.random.uniform(subkey, (in_feat, out_feat), minval=-rnd_range, maxval=rnd_range)
	biases = None
	if bias:
		biases = jnp.zeros((1, out_feat))
	return rng, weights, biases

def get_ff_block_params(rng, in_feat, out_feat, rate):
	rng, w1, b1 = get_linear_params(rng, in_feat, out_feat, bias=True)
	rng, w2, b2 = get_linear_params(rng, out_feat, out_feat, bias=True)
	rng, _ = jax.random.split(rng)
	return rng, {'W1':w1, 'W2':w2, 'b1':b1, 'b2':b2, 'rate':rate}

def get_mha_params(rng, dk, dv, out_features, num_heads, rate):
	rng, WQs, _ = get_linear_params(rng, out_features, num_heads * dk, bias=False)
	rng, WKs, _ = get_linear_params(rng, out_features, num_heads * dk, bias=False)
	rng, WVs, _ = get_linear_params(rng, out_features, num_heads * dv, bias=False)
	rng, Wout, _ = get_linear_params(rng, num_heads * dv, out_features, bias=False)
	rng, _ = jax.random.split(rng)
	return rng, {'WQs':WQs, 'WKs':WKs, 'WVs':WVs, 'Wout':Wout, 'num_heads':num_heads, 'rate':rate}

def get_ln_params(seq_len, eps=1e-9):
	gamma = jnp.ones((seq_len, 1))
	beta = jnp.zeros((seq_len, 1))
	mov_mean = jnp.zeros((seq_len, 1))
	mov_var = jnp.ones((seq_len, 1))
	return {'gamma':gamma, 'beta':beta, 'mov_mean':mov_mean, 'mov_var':mov_var, 'eps':eps}

def get_transformer_params(rng, seq_len, dk, dv, hid_size, num_heads, num_layers, ff_out, rate_att=0.2, rate_ff=0.2, eps=1e-9):
	params = {}
	for i in range(num_layers):
		params[i] = {}
		rng, params_mha_enc = get_mha_params(rng, dk, dv, hid_size, num_heads, rate_att)
		rng, params_ff_block_enc = get_ff_block_params(rng, hid_size, ff_out, rate_ff)
		params_ln_enc_1 = get_ln_params(seq_len, eps=eps)
		params_ln_enc_2 = get_ln_params(seq_len, eps=eps)
		params[i]['encoder'] = {
			'mha':params_mha_enc,
			'ff_block':params_ff_block_enc,
			'ln_1':params_ln_enc_1,
			'ln_2':params_ln_enc_2
		}

		rng, params_mha_dec_1 = get_mha_params(rng, dk, dv, hid_size, num_heads, rate_att)
		rng, params_mha_dec_2 = get_mha_params(rng, hid_size, hid_size, hid_size, num_heads, rate_att)
		rng, params_ff_block_dec = get_ff_block_params(rng, hid_size, ff_out, rate_ff)
		params_ln_dec_1 = get_ln_params(seq_len, eps=eps)
		params_ln_dec_2 = get_ln_params(seq_len, eps=eps)
		params_ln_dec_3 = get_ln_params(seq_len, eps=eps)

		params[i]['decoder'] = {
			'mha_1':params_mha_dec_1,
			'mha_2':params_mha_dec_2,
			'ff_block':params_ff_block_dec,
			'ln_1':params_ln_dec_1,
			'ln_2':params_ln_dec_2,
			'ln_3':params_ln_dec_3,
		}
	return params


	

