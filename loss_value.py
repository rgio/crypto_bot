from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports of general code
import numpy as np
import tensorflow as tf
import pdb

# Model parameters
transaction_cost = 0.0025 # 0.25% commission fee for each transaction

# Functions related to calculating the portfolio value and loss functions

def calc_minus_log_rate_return(price_change, weights, init_weights, batch_size):
	rate_return = tf.reduce_sum(tf.multiply(price_change, weights), axis=1)
	# rate_return = tf.reduce_sum(tf.multiply(tf.nn.softmax(price_change), weights), axis=1)
	# rate_return = calc_portfolio_value_change(price_change, weights, init_weights, batch_size)
	log_rate_return = tf.log(rate_return)
	portfolio_value = tf.reduce_mean(log_rate_return)
	portfolio_value = tf.scalar_mul(-1.0, portfolio_value)

	return portfolio_value

def calc_portfolio_value_change(price_change, weights, init_weights, batch_size):
	rate_return = tf.reduce_sum(tf.multiply(price_change, weights), axis=1)
	transaction_decay = calc_transaction_decay(price_change, weights, init_weights, batch_size)
	rate_return = tf.multiply(rate_return, transaction_decay)

	return rate_return

def calc_transaction_decay(price_change, weights, init_weights, batch_size):
	weights_prime = tf.multiply(price_change, init_weights)
	norm = tf.square(tf.norm(tf.sqrt(weights_prime), axis=1))
	tmp = tf.constant(weights.get_shape().as_list()[1], shape=[1])
	tiled_norm = tf.tile(norm, tmp)
	tiled_norm = tf.reshape(tiled_norm, [batch_size, weights.get_shape().as_list()[1]])
	weights_prime = tf.divide(weights_prime, tiled_norm)
	total_transaction_cost = tf.reduce_sum(tf.abs(tf.subtract(weights, weights_prime)), axis=1)
	total_transaction_cost = tf.scalar_mul(transaction_cost, total_transaction_cost)
	idendity = tf.ones(tf.shape(total_transaction_cost), dtype=tf.float32)
	transaction_decay = tf.subtract(idendity, total_transaction_cost)

	return transaction_decay
