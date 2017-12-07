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

def calc_minus_log_rate_return(price_change, weights):
	# updated_weights = tf.multiply(price_change, weights) 
	# updated_weights = tf.scalar_mul(tf.tensordot(price_change, weights), updated_weights)
	# transaction_factor = tf.fill(tf.shape(weights)[0], 1.0)
	# transaction_factor = tf.constant(1.0)
	rate_return = tf.reduce_sum(tf.multiply(price_change, weights), axis=1)
	log_rate_return = tf.log(rate_return)
	portfolio_value = tf.reduce_mean(log_rate_return)
	portfolio_value = tf.scalar_mul(-1.0, portfolio_value)
	#pdb.set_trace()

	return portfolio_value

def calc_portfolio_value_change(price_change, weights):
	rate_return = tf.reduce_sum(tf.multiply(price_change, weights), axis=1)
<<<<<<< HEAD
	return rate_return


=======
	return rate_return
>>>>>>> 16c4f5987660bc70f5d4641e4442716819a35318
