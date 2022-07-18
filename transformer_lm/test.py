def test():
	num_heads = 2
	seq_len = 4
	dk = 4
	dv = 4
	hid_size = 4
	#in_feats = 128
	bs = 2
	rnd_range = 1 / hid_size ** 0.5
	rng = jax.random.PRNGKey(42)
	np.random.seed(42)

	params = get_transformer_params(rng, seq_len, dk, dv, hid_size, num_heads, 1, hid_size, rate_att=0.2, rate_ff=0.2, eps=1e-9)
	params = params[0]
	mask_tmp = jnp.tril(jnp.ones((seq_len,seq_len)))
	mask_causal = mask_tmp.at[jnp.where(mask_tmp == 1)].set(0)
	mask_causal = mask_causal.at[jnp.where(mask_tmp == 0)].set(-1e9)
	#print(mask_causal)

	Q = jnp.array([[i+1] * dk for i in range(4)])

	mask = None

	inputs = [Q, Q, Q, mask]
	
	out = encoder_block(inputs, params['encoder'], training=True)
	print(out)
	print(out.shape)
	Q_dec = jnp.array([[i] * dk for i in range(4)])

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