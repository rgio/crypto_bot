from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports of general code
import numpy as np
import tensorflow as tf
import pdb

# Model parameters
transaction_cost = 0.0025 # 0.25% commission fee for each transaction
num_assets = 12

# Functions related to calculating the portfolio value and loss functions

def calc_minus_log_rate_return(price_change, weights, init_weights):
	rate_return = tf.reduce_sum(tf.multiply(price_change, weights), axis=1)
	# rate_return = calc_portfolio_value_change(price_change, weights, init_weights)
	log_rate_return = tf.log(rate_return)
	portfolio_value = tf.reduce_mean(log_rate_return)
	portfolio_value = tf.scalar_mul(-1.0, portfolio_value)
	#pdb.set_trace()

	return portfolio_value

def calc_portfolio_value_change(price_change, weights, init_weights):
	rate_return = tf.reduce_sum(tf.multiply(price_change, weights), axis=1)
	transaction_decay, norm = calc_transaction_decay(price_change, weights, init_weights)
	# rate_return = tf.multiply(rate_return, transaction_decay)

	return rate_return, transaction_decay, norm
	# return rate_return

def calc_transaction_decay(price_change, weights, init_weights):
	print('Shape of price_change = ', price_change.get_shape())
	print('Shape of weights = ', weights.get_shape())
	print('Shape of init_weights = ', init_weights.get_shape())
	weights_prime = tf.multiply(price_change, init_weights)
	print('Shape of weights_prime = ', weights_prime.get_shape())
	norm = tf.square(tf.norm(tf.sqrt(weights_prime), axis=1))
	print('Shape of norm = ' , norm.get_shape())
	tmp = tf.constant(num_assets, shape=[1])
	print('Shape of tmp = ', tmp.get_shape())
	tiled_norm = tf.tile(norm, tmp)
	print('Shape of tiled_norm = ', tiled_norm.get_shape())
	tiled_norm = tf.reshape(tiled_norm, [200, num_assets])
	# norm = tf.reshape(tf.tile(norm, tmp), 
	#  	[tf.shape(weights_prime)[1], tf.shape(weights_prime)[0]])
	print('Shape of reshaped tiled_norm = ', tiled_norm.get_shape())
	weights_prime = tf.divide(weights_prime, tiled_norm)
	print('Shape of normalized weights_prime = ', weights_prime.get_shape())
	total_transaction_cost = tf.reduce_sum(tf.abs(tf.subtract(weights, weights_prime)), axis=1)
	print('Shape of total_transaction_cost = ', total_transaction_cost.get_shape())
	total_transaction_cost = tf.scalar_mul(transaction_cost, total_transaction_cost)
	print('Shape of final total_transaction_cost = ', total_transaction_cost.get_shape())
	idendity = tf.ones(tf.shape(total_transaction_cost), dtype=tf.float32)
	print('Shape of idendity = ', idendity.get_shape())
	transaction_decay = tf.subtract(idendity, total_transaction_cost)
	print('Shape of returned transaction_decay = ', transaction_decay.get_shape())

	return transaction_decay, norm
