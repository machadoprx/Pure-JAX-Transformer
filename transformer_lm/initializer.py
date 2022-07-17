import jax
import jax.numpy as jnp

def get_linear_params(rng, in_feat, out_feat, bias=False):
	rnd_range = 1 / in_feat ** 0.5
	weights = jax.random.uniform(rng, (in_feat, out_feat), minval=-rnd_range, maxval=rnd_range)
	biases = None
	if bias:
		biases = jnp.zeros((1, out_feat))
	return weights, biases

def get_ff_block_params(rng, in_feat, out_feat, rate):
	w1, b1 = get_linear_params(rng, in_feat, out_feat, bias=True)
	w2, b2 = get_linear_params(rng, out_feat, out_feat, bias=True)
	return {'W1':w1, 'W2':w2, 'b1':b1, 'b2':b2, 'rate':rate, 'rng':rng}

def get_mha_params(rng, dk, dv, out_features, num_heads, rate):
	WQs, _ = get_linear_params(rng, out_features, num_heads * dk, bias=False)
	WKs, _ = get_linear_params(rng, out_features, num_heads * dk, bias=False)
	WVs, _ = get_linear_params(rng, out_features, num_heads * dv, bias=False)
	Wout, _ = get_linear_params(rng, num_heads * dv, out_features, bias=False)
	return {'WQs':WQs, 'WKs':WKs, 'WVs':WVs, 'Wout':Wout, 'num_heads':num_heads, 'rate':rate, 'rng':rng}

def get_ln_params(seq_len, eps=1e-9):
	gamma = jnp.ones((seq_len, 1))
	beta = jnp.zeros((seq_len, 1))
	mov_mean = jnp.zeros((seq_len, 1))
	mov_var = jnp.ones((seq_len, 1))
	return {'gamma':gamma, 'beta':beta, 'mov_mean':mov_mean, 'mov_var':mov_var, 'eps':eps}

