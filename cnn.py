from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports of general code
import numpy as np
import tensorflow as tf
import pdb

def variable_summaries(var):
	"""Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
	with tf.name_scope('summaries'):
		mean = tf.reduce_mean(var)
		tf.summary.scalar('mean', mean)
		with tf.name_scope('stddev'):
			stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
		tf.summary.scalar('stddev', stddev)
		tf.summary.scalar('max', tf.reduce_max(var))
		tf.summary.scalar('min', tf.reduce_min(var))
		tf.summary.histogram('histogram', var)

def weight_variable(shape):
	"""Create a weight variable with appropriate initialization."""
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	"""Create a bias variable with appropriate initialization."""
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def weights_and_biases(W_shape):
	""" """
	with tf.name_scope('weights'):
		W = weight_variable(W_shape)
		variable_summaries(W)
	with tf.name_scope('biases'):
		b = weight_variable(W_shape[-1:])
		variable_summaries(b)
	return W, b

# TODO: have no activation function possible
def fc_layer(input_tensor, W_fc_shape, layer_name, act=tf.nn.relu):
	""" """
	with tf.name_scope(layer_name):
		W_fc, b_fc = weights_and_biases(W_fc_shape)
		fc_layer = act(tf.matmul(input_tensor, W_fc) + b_fc)
	return fc_layer

def normed_fc_layer(input_tensor, W_fc_shape, layer_name, act=tf.nn.relu):
	""" """
	return tf.contrib.layers.layer_norm(fc_layer(input_tensor, W_fc_shape, layer_name, act))

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def separable_conv2d(x, W, P):
	return tf.nn.separable_conv2d(x, W, P, strides=[1, 1, 1, 1], padding='VALID')

# TODO: have no activation function possible
def conv_layer(input_tensor, W_conv_shape, layer_name, sep=False, act=tf.nn.relu):
	"""" """
	with tf.name_scope(layer_name):
		with tf.name_scope('convs'):
			W_conv, b_conv = weights_and_biases(W_conv_shape)
			if sep:
				with tf.name_scope('sep_convs'):
					with tf.name_scope('weights'):
						P_conv = weight_variable([1, 1, W_conv_shape[3]*W_conv_shape[2], W_conv_shape[3]])
						variable_summaries(P_conv)
					h_conv = act(separable_conv2d(input_tensor, W_conv, P_conv) + b_conv)
					tf.summary.histogram('separable_conv_results', h_conv)
			else:
				with tf.name_scope('standard_convs'):
					h_conv = act(conv2d(input_tensor, W_conv) + b_conv)
					tf.summary.histogram('conv_results', h_conv)
	return h_conv

def normed_conv_layer(input_tensor, W_conv_shape, layer_name, sep=False, act=tf.nn.relu):
	""" """
	return tf.contrib.layers.layer_norm(conv_layer(input_tensor, W_conv_shape, layer_name, sep, act))

def cnn_model(x, init_weights, hparams, params):
	"""Low level model for a CNN."""

	# Reshape the input to use as our first feature layer
	with tf.name_scope('price_window'):
		input_price = tf.reshape(x, [-1, params.num_coins, hparams.window_size, params.num_input_channels])

	# First convolution layer
	W_conv_shape = [1, hparams.len_conv1_filters, params.num_input_channels, hparams.num_conv1_features]
	with tf.name_scope('conv1'):
		conv_layer_1 = normed_conv_layer(input_price, W_conv_shape,'conv1', sep=hparams.conv_layers_separable)

	# Second convolution layer
	W_conv_shape = [1, hparams.window_size-hparams.len_conv1_filters+1, hparams.num_conv1_features, hparams.num_conv2_features]
	with tf.name_scope('conv2'):
		conv_layer_2 = normed_conv_layer(conv_layer_1, W_conv_shape,'conv2', sep=hparams.conv_layers_separable)

	# Add in previous weights as a feature
	with tf.name_scope('previous_weights'):
		past_weights = tf.reshape(init_weights[:,1:], [-1, params.num_coins, 1, 1])
		h_conv2_weights = tf.concat([conv_layer_2, past_weights], axis=3)

	# Dropout on 2nd convolution layer during training
	with tf.name_scope('dropout'):
		keep_prob = tf.placeholder(tf.float32)
		tf.summary.scalar('dropout_keep_prob', keep_prob)
		h_conv2_weights_dropout = tf.nn.dropout(h_conv2_weights, keep_prob)

	# Three possible endings for this cnn model: third_conv_layer, one_fc_layer, two_fc_layers
	if hparams.model_ending == 'third_conv_layer': 
		# Third and final convolution layer
		W_conv_shape = [1, params.len_conv3_filters, hparams.num_conv2_features+1, 1]
		conv_layer_3 = normed_conv_layer(h_conv2_weights_dropout, W_conv_shape, 'conv3', sep=hparams.conv_layers_separable)
		final_layer = tf.reshape(conv_layer_3, [-1, params.num_coins])
	
	else:
		# Flatten the 2nd convolution layer prior to the fully connected layers
		h_conv2_weights_dropout_flat = tf.reshape(h_conv2_weights_dropout, [-1, params.num_coins*(hparams.num_conv2_features+1)])

		if hparams.model_ending == 'one_fc_layer':
			# First and only fully connected layer
			W_fc_shape = [params.num_coins*(hparams.num_conv2_features+1), params.num_coins]
			with tf.name_scope('fc1'):
				final_layer = normed_fc_layer(h_conv2_weights_dropout_flat, W_fc_shape, 'fc1')		 
		
		elif hparams.model_ending == 'two_fc_layers': 
			# First fully connected layer
			W_fc_shape = [params.num_coins*(hparams.num_conv2_features+1), hparams.num_fc1_neurons]
			with tf.name_scope('fc1'):
				fc1_layer = normed_fc_layer(h_conv2_weights_dropout_flat, W_fc_shape, 'fc1')
			# Second and last fully connected layer
			W_fc_shape = [hparams.num_fc1_neurons, params.num_coins]
			with tf.name_scope('fc2'):
				final_layer = normed_fc_layer(fc1_layer, W_fc_shape, 'fc2')

	# Add in a bias for cash to the final layer before taking softmax to get portfolio weights
	with tf.name_scope('cash_bias'): 
		cash_bias = tf.Variable(0.0)
		cash_bias_tensor = tf.fill([tf.shape(input_price)[0], 1], cash_bias)
		final_layer_cash = tf.concat([final_layer, cash_bias_tensor], 1)
	
	# Final portfolio weight tensor
	with tf.name_scope('weights'):
		weights = tf.nn.softmax(final_layer_cash, name='output_tensor')

	return weights, keep_prob


