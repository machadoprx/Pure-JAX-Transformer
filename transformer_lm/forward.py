import jax.numpy as jnp
from jax.nn import softmax
from layers import *
from tranformer_modules import *

def forward_transformer(inputs, params, training=True):
	
	inputs_enc, inputs_dec = inputs
	seq_len = len(inputs_enc)
	n_layers = len(params['encoder'].keys())
	hid_size = params['encoder'][0]['ff_block']['W1'].shape[0]
	pos_enc = get_sinusoid_encoding_table(seq_len, hid_size, padding_idx=None)
	mask = get_causal_mask(seq_len)

	embedding_enc = embed([inputs_enc, pos_enc], params['embed'])
	out_enc = embedding_enc
	for i in range(n_layers):
		out_enc = encoder_block([out_enc, out_enc, out_enc, None], params['encoder'][i], training=training)

	embedding_dec = embed([inputs_dec, pos_enc], params['embed'])
	out_dec = embedding_dec
	for i in range(n_layers):
		out_dec = decoder_block([out_dec, out_dec, out_dec, out_enc, mask], params['decoder'][i], training=training)

	out = jnp.matmul(out_dec, params['embed']['W'])

	return out
