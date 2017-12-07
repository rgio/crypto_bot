from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports of general code
import tempfile
import numpy as np
import tensorflow as tf

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

# Imports of our own code
from price_data import *
from cnn import *
from loss_value import *
import pdb

# Set up logging
tf.logging.set_verbosity(tf.logging.INFO)

# Model hyperparameters
window_size = 50
stride = 1
transaction_cost = 0.0025 # 0.25% commission fee for each transaction
price_batch_size = 200
num_training_steps = 30000
num_coins = 11
num_input_channels = 3 # high,open, volume

def main():
	# Load training and eval data
<<<<<<< HEAD
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

	# Create the model
	#input_prices = tf.placeholder(tf.float32, [None, num_coins, window_size])
	input_prices = tf.placeholder(tf.float32, [None, num_coins, window_size, num_input_channels])
	labels = tf.placeholder(tf.float32, [None, num_coins+1])

	# Build the graph
	weights, keep_prob = new_cnn_model(input_prices)

	#pdb.set_trace()

	# Define the loss
	with tf.name_scope('loss'):
		loss = calc_minus_log_rate_return(labels, weights)
	loss = tf.reduce_mean(loss)

	# Define the optimizer
	with tf.name_scope('adam_optimizer'):
		train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

	# Define the testing conditions
	with tf.name_scope('value'):
		value = calc_portfolio_value_change(labels, weights)
	value = tf.reduce_prod(value)
	# accuracy = tf.reduce_mean(correct_prediction)

	# Decide where the graph is stored


	# Decide where the graph and model is stored
	graph_location = tempfile.mkdtemp()
	print('Saving graph to %s' % graph_location)
	train_writer = tf.summary.FileWriter(graph_location)
	train_writer.add_graph(tf.get_default_graph())
	saver = tf.train.Saver()

	# Run the training and testing
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for i in range(num_training_steps):
			batch = get_next_price_batch(train_data, train_labels, price_batch_size, num_coins, i) 
			if i % 1000 == 0:
				train_value = final_value.eval(feed_dict={
	 				input_prices: batch[0], labels: batch[1], keep_prob: 1.0})
				print('step %d, train_value %g' % (i, train_value))
	 			#print(train_value)
			train_step.run(feed_dict={input_prices: batch[0], labels: batch[1], keep_prob: 0.5})
		print('size of validation %d - validation portolio value multiplier %g' % (validation_size, final_value.eval(feed_dict={
	 		input_prices: validation_data, labels: validation_labels, keep_prob: 1.0})))
		print('size of test %d - test portolio value multiplier %g' % (test_size, final_value.eval(feed_dict={
	 		input_prices: test_data, labels: test_labels, keep_prob: 1.0})))
		save_path = saver.save(sess, "/tmp/crypto_bot_test/cnn_model.ckpt")
		print("Model saved in file: %s" % save_path)


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
