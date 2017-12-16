from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports of general code
import time
import numpy as np
import tensorflow as tf
import pdb

# Imports of our own code
from price_data import *
from cnn import *
from loss_value import *

# Main function to be used for live trading
def main():
	# Current times
	start_time = time.time()

	# Start from scratch
	tf.reset_default_graph()

	# Saver object 
	saver = tf.train.Saver()

	with tf.Session() as sess:
		# Restore model from disk.
		saver.restore(sess, '/tmp/model.cpkt')
		print('Model restored.')
		input_weights = np.zeros(12)
		input_weights[0] = 1.0
		while True:
			data, labels = get_current_window()
			weights = weights.eval(feed_dict={input_prices: data, labels: labels, 
				init_weights: input_weights, keep_prob: 1.0})
			portfolio_value = get_new_portfolio(weights)
			input_weights = weights
			time.sleep(1800)

