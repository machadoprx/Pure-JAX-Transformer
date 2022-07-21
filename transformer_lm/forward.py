import jax.numpy as jnp
from layers import *
from tranformer_modules import *

def forward_transformer(inputs, params, training=True):
	
	input, mask_input, target, mask_target = inputs

	#print(input[0], target[0])
	#quit()

	seq_len = len(input)
	n_layers = params['num_layers']
	hid_size = params['encoder_0_ff_block']['W1'].shape[0]
	pos_enc = get_sinusoid_encoding_table(seq_len, hid_size, padding_idx=None)

	embedding_enc = embed([input, pos_enc], params) * jnp.sqrt(hid_size)
	out_enc = embedding_enc
	for i in range(n_layers):
		out_enc = encoder_block([out_enc, out_enc, out_enc, mask_input], params, i, training=training)
	
	embedding_dec = embed([target, pos_enc], params) * jnp.sqrt(hid_size)
	out_dec = embedding_dec
	for i in range(n_layers):
		out_dec = decoder_block([out_dec, out_dec, out_dec, out_enc, mask_target], params, i, training=training)

	out = jnp.matmul(out_dec, params['embed']['W'])

	#print(input)
	#print(target)
	print(jnp.argmax(out, axis=-1))
	return out
