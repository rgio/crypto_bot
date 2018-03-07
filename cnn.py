from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports of general code
import numpy as np
import tensorflow as tf
import pdb


def variable_summaries(var):
	with tf.name_scope('summaries'):
		mean = tf.reduce_mean(var)
		tf.summary.scalar('mean', mean)
		with tf.name_scope('stddev'):
			stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
		tf.summary.scalar('stddev', stddev)
		tf.summary.scalar('max', tf.reduce_max(var))
		tf.summary.histogram('histogram', var)


def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def separable_conv2d(x, W, P):
	return tf.nn.separable_conv2d(x, W, P, strides=[1, 1, 1, 1], padding='VALID')

def cnn_model(x, init_weights, hparams, params):
	"""Low level model for a CNN."""

	# Reshape the input to use as our first feature layer
	with tf.name_scope('reshape_input'):
		input_price = tf.reshape(x, [-1, params.num_coins, hparams.window_size, params.num_input_channels])

	# First convolution layer
	with tf.name_scope('conv1'):
		W_conv1 = weight_variable([1, hparams.len_conv1_filters, hparams.num_input_channels, hparams.num_conv1_features])
		b_conv1 = bias_variable([hparams.num_conv1_features])
		if hparams.conv_layers_separable:
			P_conv1 = weight_variable([1, 1, hparams.num_conv1_features*hparams.num_input_channels, hparams.num_conv1_features])
			h_conv1 = tf.nn.relu(separable_conv2d(input_price, W_conv1, P_conv1) + b_conv1)
		else: # use standard convolution layer
			h_conv1 = tf.nn.relu(conv2d(input_price, W_conv1) + b_conv1)
		
	# Second convolution layer
	with tf.name_scope('conv2'):
		W_conv2 = weight_variable([1, hparams.window_size-hparams.len_conv1_filters+1, hparams.num_conv1_features, hparams.num_conv2_features])
		b_conv2 = bias_variable([hparams.num_conv2_features])
		if hparams.conv_layers_separable:
			P_conv2 = weight_variable([1, 1, hparams.num_conv2_features*hparams.num_conv1_features, hparams.num_conv2_features])
			h_conv2 = tf.nn.relu(separable_conv2d(h_conv1, W_conv2, P_conv2) + b_conv2)
		else: # use standard convolution layer
			h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)

	# Add in previous weights as a feature
	past_weights = tf.reshape(init_weights[:,1:], [-1, params.num_coins, 1, 1])
	h_conv2_weights = tf.concat([h_conv2, past_weights], axis=3)

	# Dropout on 2nd convolution layer during training
	with tf.name_scope('dropout'):
	  	keep_prob = tf.placeholder(tf.float32)
	  	h_conv2_weights_dropout = tf.nn.dropout(h_conv2_weights, keep_prob)

	# Three possible endings for this cnn model: third_conv_layer, one_fc_layer, two_fc_layers
	if hparams.model_ending == 'third_conv_layer': 
		# Third and final convolution layer
		with tf.name_scope('third_conv_layer_conv3'):
			W_conv3 = weight_variable([1, hparams.len_conv3_filters, hparams.num_conv2_features+1, 1])
			b_conv3 = bias_variable([1])
			if hparams.conv_layers_separable:
				P_conv3 = weight_variable([1, 1, hparams.num_conv2_features+1, 1])
				h_conv3 = tf.nn.relu(separable_conv2d(h_conv2_weights_dropout, W_conv3, P_conv3) + b_conv3)
			else:
				h_conv3 = tf.nn.relu(conv2d(h_conv2_weights_dropout, W_conv3) + b_conv3)
			final_layer = tf.reshape(h_conv3, [-1, hparams.num_coins])
	else:
		# Flatten the 2nd convolution layer prior to the fully connected layers
		h_conv2_weights_dropout_flat = tf.reshape(h_conv2_weights_dropout, [-1, hparams.num_coins*(hparams.num_conv2_features+1)])
		if hparams.model_ending == 'one_fc_layer':
			# First and only fully connected layer
			with tf.name_scope('one_fc_layer_fc1'):
				W_fc1 = weight_variable([hparams.num_coins*(hparams.num_conv2_features+1), hparams.num_coins])
				b_fc1 = weight_variable([hparams.num_coins])
				final_layer = tf.nn.relu(tf.matmul(h_conv2_weights_dropout_flat, W_fc1) + b_fc1)		 
		elif hparams.model_ending == 'two_fc_layers': 
			# First fully connected layer
			with tf.name_scope('two_fc_layers_fc1'):
				W_fc1 = weight_variable([hparams.num_coins*(hparams.num_conv2_features+1), hparams.num_fc1_neurons])
				b_fc1 = weight_variable([hparams.num_fc1_neurons])
				h_fc1 = tf.nn.relu(tf.matmul(h_conv2_weights_dropout_flat, W_fc1) + b_fc1)
			# Second and last fully connected layer 
			with tf.name_scope('two_fc_layers_fc2'):
				W_fc2 = weight_variable([hparams.num_fc1_neurons, hparams.num_coins])
				b_fc2 = bias_variable([hparams.num_coins])
				final_layer = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

	# Add in a bias for cash to the final layer before taking softmax to get portfolio weights
	with tf.name_scope('cash'): 
		cash_bias = tf.Variable(0.0)
		cash_bias_tensor = tf.fill([tf.shape(input_price)[0], 1], cash_bias)
		final_layer_cash = tf.concat([final_layer, cash_bias_tensor], 1)
	
	# Final portfolio weight tensor
	with tf.name_scope('weights'):
	 	weights = tf.nn.softmax(final_layer_cash, name="output_tensor")

	return weights, keep_prob


