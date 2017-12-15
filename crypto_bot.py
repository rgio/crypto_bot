from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports of general code
import datetime
import tempfile
import numpy as np
import tensorflow as tf

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

# Imports of our own code
from price_data import *
from cnn import *
from loss_value import *
from print_results import *
import pdb

# Set up logging
tf.logging.set_verbosity(tf.logging.INFO)

# Model hyperparameters
window_size = 50
stride = 1
price_batch_size = 200
num_training_steps = 400000
num_coins = 11
num_input_channels = 4 # high,open, volume, dp/dt

def main():
	# Load training and eval data
	input_array = read_data()
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

	#pdb.set_trace()

	# Create the model
	#input_prices = tf.placeholder(tf.float32, [None, num_coins, window_size])
	input_prices = tf.placeholder(tf.float32, [None, num_coins, window_size, num_input_channels])
	labels = tf.placeholder(tf.float32, [None, num_coins+1])
	init_weights = tf.placeholder(tf.float32, [None, num_coins+1])

	# Build the graph
	weights, keep_prob = new_cnn_model(input_prices, init_weights)

	# Define the loss
	with tf.name_scope('loss'):
		loss = calc_minus_log_rate_return(labels, weights, init_weights)
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
		value, decay, norm = calc_portfolio_value_change(labels, weights, init_weights)
	final_value = tf.multiply(value, decay)
	final_value = tf.reduce_prod(final_value)

	# Decide where the graph and model is stored
	graph_location = tempfile.mkdtemp()
	print('Saving graph to %s' % graph_location)
	train_writer = tf.summary.FileWriter(graph_location)
	train_writer.add_graph(tf.get_default_graph())
	saver = tf.train.Saver()
	timestamp = '{:%Y-%m-%d-%H:%M:%S}'.format(datetime.datetime.now())
	path_to_model_dir = '/tmp/crypto_bot/cnn_model_' + timestamp + '/'
	path_to_model = path_to_model_dir + 'cnn_model.ckpt'

	# Random weights for 1st training step
	random_weights = np.random.rand(price_batch_size, num_coins+1)

	# Run the training and testing
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		batch = get_next_price_batch(train_data, train_labels, price_batch_size, num_coins, 0)
		input_weights = weights.eval(feed_dict={input_prices: batch[0], labels: batch[1], 
				init_weights: random_weights, keep_prob: 1.0}) 
		for i in range(1,num_training_steps):
			batch = get_next_price_batch(train_data, train_labels, price_batch_size, num_coins, i)
			input_weights_batch = get_next_price_batch(train_data, train_labels, price_batch_size, num_coins, i-1)
			input_weights = weights.eval(feed_dict={input_prices: input_weights_batch[0], labels: input_weights_batch[1], 
				init_weights: input_weights, keep_prob: 1.0}) 
			if i % 1000 == 0:
				train_value = final_value.eval(feed_dict={input_prices: batch[0], labels: batch[1], 
					init_weights: input_weights, keep_prob: 1.0})
				train_accuracy = accuracy.eval(feed_dict={input_prices: batch[0], labels: batch[1], 
					init_weights: input_weights, keep_prob: 1.0})
				print('step %d, train_accuracy %g, train_value %g' % (i, train_accuracy, train_value))
			train_step.run(feed_dict={input_prices: batch[0], labels: batch[1], 
				init_weights: input_weights, keep_prob: 0.5})


		# Save the results
		save_path = saver.save(sess, path_to_model)
		

		# Print the results
		# input_weights = weights.eval(feed_dict={input_prices: validation_data, labels: validation_labels, 
		# 		init_weights: random_weights, keep_prob: 1.0})
		print('The accuracy on the validation set is %g' % accuracy.eval(feed_dict={input_prices: validation_data, 
			labels: validation_labels, keep_prob: 1.0}))
		# final_pvm = final_value.eval(feed_dict={input_prices: validation_data, labels: validation_labels, 
		# 	init_weights = , keep_prob: 1.0})
		pvm = value.eval(feed_dict={input_prices: validation_data, labels: validation_labels, 
			keep_prob: 1.0})
		portfolio_weights = weights.eval(feed_dict={input_prices: validation_data, labels: validation_labels, 
			keep_prob: 1.0})
		print_model_results(final_pvm, pvm, portfolio_weights, path_to_model_dir)
		print('The accuracy on the test set is %g' % accuracy.eval(feed_dict={input_prices: test_data, 
			labels: test_labels, keep_prob: 1.0}))
		final_pvm = final_value.eval(feed_dict={input_prices: test_data, labels: test_labels, 
			keep_prob: 1.0})
		pvm = value.eval(feed_dict={input_prices: test_data, labels: test_labels, keep_prob: 1.0})
		portfolio_weights = weights.eval(feed_dict={input_prices: test_data, labels: test_labels, 
			keep_prob: 1.0})
		np.savetxt('test_labels.dat', test_labels, fmt='%.8f', delimiter=' ')
		print_model_results(final_pvm, pvm, portfolio_weights, path_to_model_dir)


	# # Create the estimator
	# classifieriCNN = tf.estimator.Estimator(
	# 	model_fn = icnn_model_fn, model_dir = "/tmp/icnn_model"
	# )
	# # Training for the new CNN model
	# train_input_fn = tf.estimator.inputs.numpy_input_fn(
	# 	x = {"x": train_data},
	# 	y = train_labels,
	# 	batch_size = price_batch_size,
	# 	num_epochs = None,
	# 	shuffle=True)
	# classifieriCNN.train(
	# 	input_fn = train_input_fn,
	# 	steps = num_training_steps)
	# # Evaluate the model and print results
	# eval_input_fn = tf.estimator.inputs.numpy_input_fn(
	# 	x = {"x": test_data},
	# 	y = test_labels,
	# 	num_epochs = 1,
	# 	shuffle = False)
	# eval_results = classifieriCNN.evaluate(input_fn=eval_input_fn)
	# print(eval_results)
