from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports of general code
import os
import datetime
import tempfile
import numpy as np
import tensorflow as tf
import sys

# Imports of our own code
from price_data import *
from cnn import *
from loss_value import *
from print_results import *
import pdb

# Model hyperparameters
window_size = 50
stride = 1
price_batch_size = 100
num_training_steps = 300000
num_coins = 11
num_input_channels = 5 # high,open, volume, dp/dt

# Data fetch parameters
coins = ['eth', 'ltc', 'xrp', 'etc', 'xem', 'dash', 'steem', 'bts', 'strat', 'xmr', 'zec',]
data_dir = 'data/test/'

def main():
	# Load training and eval data
	input_array = read_data(coins, data_dir)
	total_time_steps = input_array.shape[1]
	train_size = int(total_time_steps*0.7)
	validation_size = int(total_time_steps*0.15)
	test_size = int(total_time_steps*0.15)
	train, validation, test = split_data(input_array, train_size, validation_size, test_size)
	train_data, train_labels = get_data(train, window_size, stride)
	validation_data, validation_labels = get_data(validation, window_size, stride)
	validation_labels = np.reshape(validation_labels, (validation_labels.shape[0], num_coins))
	btc_btc = np.ones( (1, validation_labels.shape[0]), dtype=np.float32)
	validation_labels = np.insert(validation_labels, 0, btc_btc, axis=1)
	test_data, test_labels = get_data(test, window_size, stride)
	test_labels = np.reshape(test_labels, (test_labels.shape[0], num_coins))
	btc_btc = np.ones( (1, test_labels.shape[0]), dtype=np.float32)
	test_labels = np.insert(test_labels, 0, btc_btc, axis=1)

	# realtime_input = read_data('realtime_data/')
	# realtime_size = realtime_input.shape[1]
	# realtime_test,realtime_test_labels = get_data(realtime_input,window_size,stride)
	# realtime_test_labels = np.reshape(realtime_test_labels, (realtime_test_labels.shape[0], num_coins))
	# btc_btc = np.ones( (1, realtime_test_labels.shape[0]), dtype=np.float32)
	# realtime_test_labels = np.insert(realtime_test_labels, 0, btc_btc, axis=1)

	# Create the model
	input_prices = tf.placeholder(tf.float32, [None, num_coins, window_size, num_input_channels])
	labels = tf.placeholder(tf.float32, [None, num_coins+1])
	init_weights = tf.placeholder(tf.float32, [None, num_coins+1])
	batch_size = tf.placeholder(tf.int32)

	# Build the graph
	weights, keep_prob = new_cnn_model(input_prices, init_weights)

	# Define the loss
	with tf.name_scope('loss'):
		loss = calc_minus_log_rate_return(labels, weights, init_weights, batch_size)
	loss = tf.reduce_mean(loss)

	# Define the optimizer
	with tf.name_scope('adam_optimizer'):
		train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

	# Define the accuracy of the model
	with tf.name_scope('accuracy'):
		correct_prediction = tf.equal(tf.argmax(weights, axis=1), tf.argmax(labels, axis=1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	# Define the testing conditions
	with tf.name_scope('value'):
		value = calc_portfolio_value_change(labels, weights, init_weights, batch_size)
	final_value = tf.reduce_prod(value)

	# Decide where the graph and model is stored
	timestamp = '{:%Y-%m-%d-%H:%M:%S}'.format(datetime.datetime.now())
	home_path = os.path.join(r'/tmp/crypto_bot', timestamp)
	path_to_model_dir = os.path.join(home_path, 'cnn_model')
	os.makedirs(path_to_model_dir)
	path_to_model = os.path.join(path_to_model_dir, 'cnn_model.ckpt')	
	graph_location = os.path.join(home_path, 'graph')
	os.makedirs(graph_location)
	print('Saving graph to %s' % graph_location)
	train_writer = tf.summary.FileWriter(graph_location)
	train_writer.add_graph(tf.get_default_graph())
	saver = tf.train.Saver()

	# Random weights for 1st training step
	random_weights = np.random.rand(price_batch_size, num_coins+1)

	steps_per_epoch = (train_data.shape[0]-price_batch_size)
	memory_array = np.random.rand(steps_per_epoch, num_coins+1)
	#pdb.set_trace()

	# Run the training and testing
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		batch = get_next_price_batch(train_data, train_labels, price_batch_size, num_coins, 0)
		input_weights = weights.eval(feed_dict={input_prices: batch[0], labels: batch[1], 
				init_weights: memory_array[:price_batch_size], batch_size: price_batch_size, keep_prob: 1.0})

		for i in range(1,num_training_steps):
			batch = get_next_price_batch(train_data, train_labels, price_batch_size, num_coins, i)
			input_weights_batch = get_next_price_batch(train_data, train_labels, price_batch_size, num_coins, i-1)
			current_time_step = input_weights_batch[2]
			input_weights = weights.eval(feed_dict={input_prices: input_weights_batch[0], labels: input_weights_batch[1], 
				init_weights: input_weights, batch_size: price_batch_size, keep_prob: 1.0})  
			memory_array[current_time_step:current_time_step+price_batch_size]=input_weights
			#pdb.set_trace()
			if i % 1000 == 0:
				train_value = final_value.eval(feed_dict={input_prices: batch[0], labels: batch[1], 
					init_weights: input_weights, batch_size: price_batch_size, keep_prob: 1.0})
				train_accuracy = accuracy.eval(feed_dict={input_prices: batch[0], labels: batch[1], 
					init_weights: input_weights, batch_size: price_batch_size, keep_prob: 1.0})
				print('step %d, train_accuracy %g, train_value %g' % (i, train_accuracy, train_value))
			train_step.run(feed_dict={input_prices: batch[0], labels: batch[1], 
				init_weights: input_weights, batch_size: price_batch_size, keep_prob: 0.5})

		# Save the results
		save_path = saver.save(sess, path_to_model)
		print('The weights are saved in %s' % path_to_model)

		# Print the validation results
		random_weights = np.random.rand(validation_labels.shape[0], validation_labels.shape[1])
		input_weights = weights.eval(feed_dict={input_prices: validation_data, labels: validation_labels, 
		 		init_weights: random_weights, batch_size: validation_labels.shape[0], keep_prob: 1.0})
		print('The accuracy on the validation set is %g' % accuracy.eval(feed_dict={input_prices: validation_data, 
			labels: validation_labels, init_weights: input_weights, batch_size: validation_labels.shape[0], keep_prob: 1.0}))
		final_pvm = final_value.eval(feed_dict={input_prices: validation_data, labels: validation_labels, 
		 	init_weights: input_weights, batch_size: validation_labels.shape[0], keep_prob: 1.0})
		pvm = value.eval(feed_dict={input_prices: validation_data, labels: validation_labels, 
			init_weights: input_weights, batch_size: validation_labels.shape[0], keep_prob: 1.0})
		portfolio_weights = weights.eval(feed_dict={input_prices: validation_data, labels: validation_labels, 
			init_weights: input_weights, batch_size: validation_labels.shape[0], keep_prob: 1.0})
		print_model_results(final_pvm, pvm, portfolio_weights, path_to_model_dir)
		
		# Print the test results
		random_weights = np.random.rand(test_labels.shape[0], test_labels.shape[1])
		input_weights = weights.eval(feed_dict={input_prices: test_data, labels: test_labels, 
		 		init_weights: random_weights, batch_size: test_labels.shape[0], keep_prob: 1.0})
		print('The accuracy on the test set is %g' % accuracy.eval(feed_dict={input_prices: test_data, 
			labels: test_labels, init_weights: input_weights, batch_size: test_labels.shape[0], keep_prob: 1.0}))
		final_pvm = final_value.eval(feed_dict={input_prices: test_data, labels: test_labels, 
			init_weights: input_weights, batch_size: test_labels.shape[0], keep_prob: 1.0})
		pvm = value.eval(feed_dict={input_prices: test_data, labels: test_labels, 
			init_weights: input_weights, batch_size: test_labels.shape[0], keep_prob: 1.0})
		portfolio_weights = weights.eval(feed_dict={input_prices: test_data, labels: test_labels, 
			init_weights: input_weights, batch_size: test_labels.shape[0], keep_prob: 1.0})
		np.savetxt('test_labels.dat', test_labels, fmt='%.8f', delimiter=' ')
		print_model_results(final_pvm, pvm, portfolio_weights, path_to_model_dir)

if __name__=='__main__':
    main()