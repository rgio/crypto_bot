from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports of general code
import numpy as np
import tensorflow as tf
import pdb

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
		W_conv1 = weight_variable([1, 3, params.num_input_channels, hparams.num_conv1_features])
		P_conv1 = weight_variable([1, 1, hparams.num_conv1_features*params.num_input_channels, hparams.num_conv1_features])
		b_conv1 = bias_variable([hparams.num_conv1_features])
		#h_conv1 = tf.nn.relu(conv2d(input_price, W_conv1) + b_conv1)
		h_conv1 = tf.nn.relu(separable_conv2d(input_price, W_conv1, P_conv1) + b_conv1)

	# Second convolution layer
	with tf.name_scope('conv2'):
		W_conv2 = weight_variable([1, hparams.window_size-2, hparams.num_conv1_features, hparams.num_conv2_features])
		P_conv2 = weight_variable([1, 1, hparams.num_conv2_features*hparams.num_conv1_features, hparams.num_conv2_features])
		b_conv2 = bias_variable([hparams.num_conv2_features])
		#h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)
		h_conv2 = tf.nn.relu(separable_conv2d(h_conv1, W_conv2, P_conv2) + b_conv2)

	# Uncomment when using 1 x 1 convolution
	# Dropout on 2nd convolution layer during training
	# with tf.name_scope('dropout'):
	#  	keep_prob = tf.placeholder(tf.float32)
	#  	h_conv2_drop = tf.nn.dropout(h_conv2, keep_prob)
	
	# Add in previous weights as a feature
	past_weights = tf.reshape(init_weights[:,1:], [-1, params.num_coins, 1, 1])
	h_conv2_weights = tf.concat([h_conv2, past_weights], axis=3)
	# h_conv2_weights_dropout = tf.concat([h_conv2_drop, past_weights], axis=3)

	# To run with 1x1 convolution layer uncomment lines conv3 through h_conv3_flat_cash
	## 1 x 1 convolutional layer
	# with tf.name_scope('conv3'):
	# 	W_conv3 = weight_variable([1, 1, hparams.num_conv2_features+1, 1])
	# 	b_conv3 = bias_variable([1])
	# 	# h_conv3 = tf.nn.relu(conv2d(h_conv2_weights, W_conv3) + b_conv3)
	# 	h_conv3 = tf.nn.relu(conv2d(h_conv2_weights_dropout, W_conv3) + b_conv3)
	# 	h_conv3_flat = tf.reshape(h_conv3, [-1, params.num_coins])

	# with tf.name_scope('cash'): 
	# 	cash_bias = tf.Variable(0.1)
	# 	cash_bias_tensor = tf.fill([tf.shape(input_price)[0], 1], cash_bias)
	# 	h_conv3_flat_cash = tf.concat([h_conv3_flat, cash_bias_tensor], 1)			

	# To run with two fully-connected layers uncomment lines h_conv2_flat through h_fc2_cash
	# Flatten the 2nd convolution layer prior to the fully connected layers
	h_conv2_flat = tf.reshape(h_conv2_weights, [-1, params.num_coins*(hparams.num_conv2_features+1)])
	#h_conv2_flat = tf.reshape(h_conv2, [-1, params.num_coins*hparams.num_conv2_features])

	# First fully connected layer
	with tf.name_scope('fc1'):
		W_fc1 = weight_variable([params.num_coins*(hparams.num_conv2_features+1), hparams.num_fc1_neurons])
		b_fc1 = weight_variable([hparams.num_fc1_neurons])
		h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)

	# Dropout on first connected layer during training
	with tf.name_scope('dropout'):
		keep_prob = tf.placeholder(tf.float32)
		h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
	
	# Second fully connected layer - softmax of this layer is the portfolio weights
	with tf.name_scope('fc2'):
		W_fc2 = weight_variable([hparams.num_fc1_neurons, params.num_coins])
		b_fc2 = bias_variable([params.num_coins])
		h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

	# Add in a bias for cash
	with tf.name_scope('cash'): 
		cash_bias = tf.Variable(0.0)
		cash_bias_tensor = tf.fill([tf.shape(input_price)[0], 1], cash_bias)
		# h_fc2_cash = tf.concat([h_fc2, cash_bias_tensor], 1)
		h_fc2_cash = tf.concat([cash_bias_tensor, h_fc2], 1)

	# Final portfolio weight tensor
	with tf.name_scope('weights'):
		weights = tf.nn.softmax(h_fc2_cash, name="output_tensor")
		# weights = tf.nn.softmax(h_conv3_flat_cash, name="output_tensor")

	return weights, keep_prob


