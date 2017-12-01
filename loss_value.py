from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports of general code
import numpy as np
import tensorflow as tf


# Functions related to calculating the portfolio value and loss functions

def calc_minus_log_rate_return(price_change, weights):
	# updated_weights = tf.multiply(price_change, weights) 
	# updated_weights = tf.scalar_mul(tf.tensordot(price_change, weights), updated_weights)
	# transaction_factor = tf.fill(tf.shape(weights)[0], 1.0)
	transaction_factor = tf.constant(1.0)
	rate_return = tf.reduce_sum(tf.multiply(price_change, weights), axis=1)
	# log_rate_return = tf.log(rate_return)
	portfolio_value = tf.reduce_mean(rate_return)
	portfolio_value = tf.scalar_mul(-1.0, portfolio_value)

	return portfolio_value

