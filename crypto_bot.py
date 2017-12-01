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

# Set up logging
tf.logging.set_verbosity(tf.logging.INFO)

# Model hyperparameters
window_size = 50
stride = 1
transaction_cost = 0.0025 # 0.25% commission fee for each transaction
price_batch_size = 100
num_training_steps = 100000
num_coins = 11

def main():
	# Load training and eval data
	global_price_array = read_data()
	total_time_steps = len(global_price_array[0])
	train_size = int(total_time_steps*0.7)
	validation_size = int(total_time_steps*0.15)
	test_size = int(total_time_steps*0.15)
	train, validation, test = split_data(global_price_array, train_size, validation_size, test_size)
	train_data, train_labels = get_data(train, window_size, stride)
	validation_data, validation_labels = get_data(validation, window_size, stride)
	test_data, test_labels = get_data(test, window_size, stride)
	test_labels = np.reshape(test_labels, (test_labels.shape[0], num_coins))
	btc_btc = np.ones( (1, test_labels.shape[0]), dtype=np.float32)
	test_labels = np.insert(test_labels, 0, btc_btc, axis=1)

	# Create the model
	input_prices = tf.placeholder(tf.float32, [None, num_coins, window_size])
	labels = tf.placeholder(tf.float32, [None, num_coins+1])

	# Build the graph
	weights, keep_prob = new_cnn_model(input_prices)

	# Define the loss
	with tf.name_scope('loss'):
		# loss = tf.losses.log_loss(tf.nn.softmax(labels), weights)
		loss = calc_minus_log_rate_return(tf.nn.softmax(labels), weights)
		# cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=tf.nn.softmax(labels), logits=weights)
	# cross_entropy = tf.reduce_mean(cross_entropy)
	loss = tf.reduce_mean(loss)

	# Define the optimizer
	with tf.name_scope('adam_optimizer'):
		train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

	# Define the testing conditions
	with tf.name_scope('value'):
		# correct_prediction = tf.equal(tf.argmax(labels, 1), tf.argmax(weights, 1))
		# correct_prediction = tf.cast(correct_prediction, tf.float32)
		# value = tf.losses.log_loss(tf.nn.softmax(labels), weights)
		value = calc_minus_log_rate_return(labels, weights)
	value = tf.reduce_mean(value)
	# accuracy = tf.reduce_mean(correct_prediction)

	# Decide where the graph is stored
	graph_location = tempfile.mkdtemp()
	print('Saving graph to %s' % graph_location)
	train_writer = tf.summary.FileWriter(graph_location)
	train_writer.add_graph(tf.get_default_graph())

	# Run the training and testing
	with tf.Session() as sess:
	 	sess.run(tf.global_variables_initializer())
	 	for i in range(num_training_steps):
	 		batch = get_next_price_batch(train_data, train_labels, price_batch_size, num_coins) 
	 		if i % 1000 == 0:
	 			train_value = value.eval(feed_dict={
	 				input_prices: batch[0], labels: batch[1], keep_prob: 1.0})
	 			print('step %d, training loss %g' % (i, train_value))
	 		train_step.run(feed_dict={input_prices: batch[0], labels: batch[1], keep_prob: 0.5})
	 	print('test loss %g' % value.eval(feed_dict={
	 		input_prices: test_data, labels: test_labels, keep_prob: 1.0}))

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




