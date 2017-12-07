from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports of general code
import numpy as np
import tensorflow as tf
import pdb

from tensorflow.contrib import learn

# Price data/ CNN specific hyperparameters
num_coins = 11
window_size = 50
filterSize = [num_coins, 4]
hiddenUnits = 500
num_filters = 12
num_input_channels = 3 # will become 3 (or more if volume taken into account)
num_conv1_features = 2
num_conv2_features = 20
num_fc1_neurons = 128

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.2)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.2, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def separable_conv2d(x, W, P):
	return tf.nn.separable_conv2d(x, W, P, strides=[1, 1, 1, 1], padding='VALID')


def new_cnn_model(x):
	"""Low level model for a CNN."""

	# Reshape the input to use as our first feature layer
	with tf.name_scope('reshape_input'):
		input_price = tf.reshape(x, [-1, num_coins, window_size, num_input_channels])

	# First convolution layer
	with tf.name_scope('conv1'):

		W_conv1 = weight_variable([1, 3, num_input_channels, num_conv1_features])
		P_conv1 = weight_variable([1,1,num_conv1_features*num_input_channels,num_conv1_features])
		b_conv1 = bias_variable([num_conv1_features])
		#h_conv1 = tf.nn.relu(conv2d(input_price, W_conv1) + b_conv1)
		h_conv1 = tf.nn.relu(separable_conv2d(input_price, W_conv1, P_conv1) + b_conv1)

	# Second convolution layer
	with tf.name_scope('conv2'):
		W_conv2 = weight_variable([1, window_size-2, num_conv1_features, num_conv2_features])
		b_conv2 = bias_variable([num_conv2_features])
		h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)
	
	# TODO: concate  weights from previous step to conv_2 (or maybe conv2_flat)
	# Add in previous weights as a feature
	# past_weights = tf.fill([tf.shape(input_price)[0], num_coins, 1, 1], 1.0/num_coins)
	# h_conv2_weights = tf.concat([h_conv2, past_weights], 3)

	# Flatten the 2nd convolution layer prior to the fully connected layers
	#h_conv2_flat = tf.reshape(h_conv2, [-1, num_coins*num_input_channels*num_conv2_features])
	h_conv2_flat = tf.reshape(h_conv2, [-1, num_coins*num_conv2_features])


	# First fully connected layer
	with tf.name_scope('fc1'):
		W_fc1 = weight_variable([num_coins*num_conv2_features, num_fc1_neurons])
		b_fc1 = weight_variable([num_fc1_neurons])
		# Flatten the 2nd convolution layer prior to the fully connected layers
		h_conv2_flat = tf.reshape(h_conv2, [-1, num_coins*num_conv2_features])	
		h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)

	# Dropout on first connected layer during training
	with tf.name_scope('dropout'):
		keep_prob = tf.placeholder(tf.float32)
		h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
	
	# Second fully connected layer - softmax of this layer is the portfolio weights
	with tf.name_scope('fc2'):
		W_fc2 = weight_variable([num_fc1_neurons, num_coins])
		b_fc2 = bias_variable([num_coins])
		h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
	

	# Add in a bias for cash
	with tf.name_scope('cash'): 
		cash_bias = tf.Variable(0.0)
		cash_bias_tensor = tf.fill([tf.shape(input_price)[0], 1], cash_bias)
		h_fc2_cash = tf.concat([h_fc2, cash_bias_tensor], 1)
		
	# Final portfolio weight tensor
	with tf.name_scope('weights'):
		weights = tf.nn.softmax(h_fc2_cash, name="output_tensor")
	# past_weights = tf.slice()

	return weights, keep_prob

def icnn_model_fn(features, labels, mode):
	"""Model function for a iCNN."""
	input_layer = tf.reshape(features["x"], [-1, num_filters, window_size, 1])
	labels = tf.reshape(labels, [-1, num_coins])
	conv1 = tf.layers.conv2d(
			inputs=input_layer,
			filters=num_filters,
			kernel_size=filterSize,
			strides=(1,1),
			padding="valid",
			activation=tf.nn.relu)
	conv1_flat = tf.reshape(conv1, [60, num_filters * 47])
	dense = tf.layers.dense(inputs=conv1_flat, units=hiddenUnits, activation=tf.nn.relu)
	dropout = tf.layers.dropout(
		inputs=dense, rate=0.4, training= (mode == learn.ModeKeys.TRAIN))
	out = tf.layers.dense(inputs=dropout, units=num_coins, activation=tf.nn.softmax)
	predictions = {
		"best_guess": tf.argmax(input=out, axis=1),
		"weights": tf.identity(out, name="porfolio_weights")
	}
	loss = tf.losses.log_loss(out, labels)
	if mode == tf.estimator.ModeKeys.TRAIN:
		optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00001)
		train_op = optimizer.minimize(
			loss=loss,
			global_step=tf.train.get_global_step())
		return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

