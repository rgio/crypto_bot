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
TIME_STAMP = False

# Helper functions
def initialize_weight(path_to_model_dir):
	hparams_dict = json.load(open(path_to_model_dir + 'hyperparameters.json'))
	hparams = tf.contrib.training.HParams(**hparams_dict)
	input_prices = np.ones((hparams.num_coins, hparams.window_size, hparams.num_input_channels), dtype=np.float32)
	input_weights = np.zeros((1,hparams.num_coins+1), dtype=np.float32)
	weights, keep_prob = cnn.cnn_model(input_prices, input_weights, hparams)
	return weights, hparams

def initialize_hparams(path_to_model_dir):
	hparams_dict = json.load(open(path_to_model_dir + 'hyperparameters.json'))
	hparams = tf.contrib.training.HParams(**hparams_dict)
	return hparams

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
	#
	#weights, hparams = initialize_weights(path_to_model_dir)
	hparams = initialize_hparams(path_to_model_dir)
	input_prices = tf.placeholder(tf.float32, [None, hparams.num_coins, hparams.window_size, hparams.num_input_channels])
	labels = tf.placeholder(tf.float32, [None, hparams.num_coins+1])
	init_weights = tf.placeholder(tf.float32, [None, hparams.num_coins+1])
	batch_size = tf.placeholder(tf.int32)
	weights, keep_prob = cnn.cnn_model(input_prices, init_weights, hparams)


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
			pdb.set_trace()
			weights = weights.eval(feed_dict={input_prices: data, labels: labels,
				init_weights: input_weights, batch_size: 1, keep_prob: 1.0})
			portfolio_value = pol.get_new_portfolio(weights)
			input_weights = weights
			print("Trade completed\n")
			time.sleep(1800)

if __name__ == '__main__':
	main()
