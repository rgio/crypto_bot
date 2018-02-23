from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports of general code
import time
import numpy as np
import tensorflow as tf
import pdb
import json

# Imports of our own code
import hparams as hp
import price_data as pdata
import cnn
import poloniex_api as pol 

# Global variable
# change this to false to use most recent training run
# change this to the switch the model that gets restored
TIME_STAMP = "2018-02-22_10-31" 

# Helper functions
def initialize_weights(path_to_model_dir):
	hparams_dict = json.load(open(path_to_model_dir + 'hyperparameters.json'))
	hparams = tf.contrib.training.HParams(**hparams_dict)
	init_prices = np.ones((hparams.num_coins, hparams.window_size, hparams.num_input_channels), dtype=np.float32)
	input_weights = np.zeros((1,hparams.num_coins+1), dtype=np.float32)
	weights = cnn.cnn_model(init_prices, input_weights, hparams)
	return weights, hparams

def get_path_to_model_dir(time_stamp=False):
	if time_stamp:
		return 'tmp/crypto_bot/cnn_model_' + time_stamp + '/'
	else:
		with open('tmp/crypto_bot/most_recent_time_stamp.txt') as f:
			time_stamp = f.read().strip()
		return 'tmp/crypto_bot/cnn_model_' + time_stamp + '/'

# Main function to be used for live trading
def main():
	# Current times
	start_time = time.time()

	# Start from scratch
	tf.reset_default_graph()

	# Create variables that will be restored
	path_to_model_dir = get_path_to_model_dir(TIME_STAMP)
	weights, hparams = initialize_weights(path_to_model_dir)

	# Saver object 
	saver = tf.train.Saver()

	with tf.Session() as sess:
		# Restore model from disk.
		path_to_ckpt_files = path_to_model_dir + 'cnn_best_model.ckpt'
		saver.restore(sess, path_to_ckpt_files) 
		print('Model restored from %s\n' %  path_to_ckpt_files)
		while True:
			data, labels = pdata.get_current_window()
			input_weights = pol.get_weights()
			weights = weights.eval(feed_dict={input_prices: data, labels: labels, 
				init_weights: input_weights, keep_prob: 1.0})
			portfolio_value = pol.get_new_portfolio(weights)
			input_weights = weights
			time.sleep(1800)
			print("ran though an iteration\n")
