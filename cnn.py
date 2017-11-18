from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports of general code
import numpy as np
import tensorflow as tf

from tensorflow.contrib import learn
# from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

# Price data/ CNN specific hyperparameters
num_coins = 12
window_size = 50
filterSize = [num_coins, 4]
hiddenUnits = 500
num_filters = 12
num_input_channels = 1 # will become 3 (or more if volume taken into account)
num_conv1_features = 2
num_conv2_features = 20
learning_rate = 0.00003 # alpha (step size) of the Adam optimization

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def cnn_model_fn(features, labels, mode):
	"""Model function for a CNN."""
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

def new_cnn_model(features, labels, mode):
	"""Low level model for a CNN."""
#	input_layer = tf.placeholder(tf.float16, shape=[None, num_coins*window_size])
	input_price = tf.reshape(features["x"], [-1, num_coins, window_size, num_input_channels])
	labels = tf.placeholder(tf.float16, shape=[None, num_coins])
	W_conv1 = weight_variable([1, 3, num_input_channels, num_conv1_features])
	b_conv1 = bias_variable([num_conv1_features])
	h_conv1 = tf.nn.relu(conv2d(input_price, W_conv1) + b_conv1)
	W_conv2 = weight_variable([1, window_size-2, num_conv1_features, num_conv2_features])
	b_conv2 = bias_variable([num_conv2_features])
	h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)
	# add in 

