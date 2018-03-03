from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import tempfile
import pathlib
import pdb

import numpy as np
import tensorflow as tf
from keras.callbacks import TensorBoard

# Imports of our own code
import hparams as hp
import price_data as pdata
import cnn
import loss_value as lv
import print_results as prnt
import functions as fn
import pdb


def main():
	hparams = hp.set_parameters()
	crypto_bot = CryptoBot(hparams)



class CryptoBot:
	def __init__(self, hparams, tuning=False, hparam_str=None): # TODO: add directory as argument and have files structured in base directory
		# Load training and eval data
		# hparams = hp.set_hyperparameters()
		if tuning:
			hyperparam_str = 'hp'
			basedir = 'tmp/tuning' + hparam_str
		else:
			timestamp = '{:%Y-%m-%d_%H-%M}'.format(datetime.datetime.now())
			basedir = 'tmp/output' + '/{timestamp}'

		callback_log = TensorBoard(
			log_dir=log_dir,
			histogram_freq=0,
			batch_size=32,
			write_graph=True,
			write_grads=False,
			write_images=False)

		try:
			input_array = pdata.read_data()
		except:
			pdata.fetch_data()
			input_array = pdata.read_data()
		total_time_steps = input_array.shape[1]
		train_size = int(total_time_steps*0.7)
		validation_size = int(total_time_steps*0.15)
		test_size = int(total_time_steps*0.15)
		train, validation, test = pdata.split_data(input_array, train_size, validation_size, test_size)
		train_data, train_labels = pdata.get_data(train, hparams.window_size, hparams.stride)
		validation_data, validation_labels = pdata.get_data(validation, hparams.window_size, hparams.stride)
		validation_labels = np.reshape(validation_labels, (validation_labels.shape[0], hparams.num_coins))
		btc_btc = np.ones( (1, validation_labels.shape[0]), dtype=np.float32)
		validation_labels = np.insert(validation_labels, 0, btc_btc, axis=1)
		validation_path = basedir + '/validation'
		fn.check_path(validation_path)
		opt_val_portfolio, opt_val_port_return = pdata.calc_optimal_portfolio(validation_labels, validation_path)
		test_data, test_labels = pdata.get_data(test, hparams.window_size, hparams.stride)
		test_labels = np.reshape(test_labels, (test_labels.shape[0], hparams.num_coins))
		btc_btc = np.ones( (1, test_labels.shape[0]), dtype=np.float32)
		test_labels = np.insert(test_labels, 0, btc_btc, axis=1)
		test_path = basedir + '/test'
		fn.check_path(test_path)
		opt_test_portfolio, opt_test_port_return = pdata.calc_optimal_portfolio(test_labels, test_path)

		# Create the model
		input_prices = tf.placeholder(tf.float32, [None, hparams.num_coins, hparams.window_size, hparams.num_input_channels])
		labels = tf.placeholder(tf.float32, [None, hparams.num_coins+1])
		init_weights = tf.placeholder(tf.float32, [None, hparams.num_coins+1])
		batch_size = tf.placeholder(tf.int32)

		# Build the graph
		weights, keep_prob = cnn.cnn_model(input_prices, init_weights, hparams)

		# Define the loss
		with tf.name_scope('loss'):
			loss = lv.calc_minus_log_rate_return(labels, weights, init_weights, batch_size)
		loss = tf.reduce_mean(loss)

		# Define the optimizer
		with tf.name_scope('adam_optimizer'):
			train_step = tf.train.AdamOptimizer(hparams.learning_rate).minimize(loss)

		# Define the accuracy of the model
		with tf.name_scope('accuracy'):
			correct_prediction = tf.equal(tf.argmax(weights, axis=1), tf.argmax(labels, axis=1))
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

		# Define the testing conditions
		with tf.name_scope('value'):
			value = lv.calc_portfolio_value_change(labels, weights, init_weights, batch_size)
		final_value = tf.reduce_prod(value)

		# Decide where the graph and model is stored
		graph_location = tempfile.mkdtemp()
		print('Saving graph to %s' % graph_location)
		train_writer = tf.summary.FileWriter(graph_location)
		train_writer.add_graph(tf.get_default_graph())
		saver = tf.train.Saver()
		timestamp = '{:%Y-%m-%d_%H-%M}'.format(datetime.datetime.now())
		timestamp_path = basedir + '/timestamps'
		fn.check_path(timestamp_path)
		timestamp_file = timestamp_path + '/timestamps.txt'
		with open(timestamp_file, 'w') as f:
			f.write('%s' % timestamp)
		path_to_model_dir = basedir + '/model'
		fn.check_path(path_to_model_dir)
		pathlib.Path(path_to_model_dir).mkdir(parents=True, exist_ok=True)
		prnt.print_hyperparameters(hparams, path_to_model_dir)
		path_to_final_model = path_to_model_dir + '/cnn_model.ckpt'
		path_to_best_model = path_to_model_dir + '/cnn_best_model.ckpt'
		best_val_value = 0.0 # used to save

		# Random weights for 1st training step
		random_weights = np.random.rand(hparams.batch_size, hparams.num_coins+1)
		val_weights = np.random.rand(validation_labels.shape[0], validation_labels.shape[1])

		# Stores our portfolio weights
		memory_array = np.random.rand(train_data.shape[0], hparams.num_coins+1)

		# Run the training and testing
		train_path = basedir + '/train'
		fn.check_path(train_path)
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			batch = pdata.get_next_price_batch(train_data, train_labels, 0, hparams)
			input_weights = weights.eval(feed_dict={input_prices: batch[0], labels: batch[1],
					init_weights: memory_array[:hparams.batch_size], batch_size: hparams.batch_size, keep_prob: 1.0})
			memory_array[:hparams.batch_size] = input_weights
			for i in range(1, hparams.num_training_steps):
				batch = pdata.get_next_price_batch(train_data, train_labels, i, hparams)
				input_weights_batch = pdata.get_specific_price_batch(train_data, train_labels, batch[2]-1, hparams)
				input_weights = weights.eval(feed_dict={input_prices: input_weights_batch[0], labels: input_weights_batch[1],
					init_weights: memory_array[batch[2]:batch[2]+hparams.batch_size], batch_size: hparams.batch_size, keep_prob: 1.0})
				memory_array[batch[2]:batch[2]+hparams.batch_size] = input_weights
				if i % 1000 == 0:
					pdata.calc_optimal_portfolio(batch[1], train_path)
					train_value = final_value.eval(feed_dict={input_prices: batch[0], labels: batch[1],
						init_weights: input_weights, batch_size: hparams.batch_size, keep_prob: 1.0})
					train_accuracy = accuracy.eval(feed_dict={input_prices: batch[0], labels: batch[1],
						init_weights: input_weights, batch_size: hparams.batch_size, keep_prob: 1.0})
					print('Step = %d\nBatch = %d\nTrain_accuracy = %g\nTrain_value = %g' % (i, batch[2], train_accuracy, train_value))
				if i % 10000 == 0:
					val_weights = weights.eval(feed_dict={input_prices: validation_data, labels: validation_labels,
							init_weights: val_weights, batch_size: validation_labels.shape[0], keep_prob: 1.0})
					final_pvm = final_value.eval(feed_dict={input_prices: validation_data, labels: validation_labels,
						init_weights: val_weights, batch_size: validation_labels.shape[0], keep_prob: 1.0})
					val_weights = weights.eval(feed_dict={input_prices: validation_data, labels: validation_labels,
						init_weights: val_weights, batch_size: validation_labels.shape[0], keep_prob: 1.0})
					print('Step %d, validation_size %d, validation_value %g' % (i, validation_labels.shape[0], final_pvm))
					if (final_pvm > best_val_value):
						save_path_best_model = saver.save(sess, path_to_best_model)
						best_val_value = final_pvm
						print('new best validation value, best model weights saved in %s\n' % save_path_best_model)
				train_step.run(feed_dict={input_prices: batch[0], labels: batch[1],
					init_weights: input_weights, batch_size: hparams.batch_size, keep_prob: hparams.dropout_keep_prob})

			# Save the results
			save_path_final_model = saver.save(sess, path_to_final_model)
			print('The final model weights are saved in %s' % save_path_final_model)

			# Print the validation results
			random_weights = np.random.rand(validation_labels.shape[0], validation_labels.shape[1])
			input_weights = weights.eval(feed_dict={input_prices: validation_data, labels: validation_labels,
					init_weights: random_weights, batch_size: validation_labels.shape[0], keep_prob: 1.0})
			print('The accuracy on the validation set is %g' % accuracy.eval(feed_dict={input_prices: validation_data,
				labels: validation_labels, init_weights: input_weights, batch_size: validation_labels.shape[0], keep_prob: 1.0}))
			self.validation_accuracy = accuracy.eval(feed_dict={
				input_prices: validation_data,
				labels: validation_labels,
				init_weights: val_weights,
				batch_size: validation_labels.shape[0],
				keep_prob: 1.0})
			print(f'The accuracy (with val_weights) on the validation set is {self.validation_accuracy}')
			final_pvm = final_value.eval(feed_dict={input_prices: validation_data, labels: validation_labels,
				init_weights: input_weights, batch_size: validation_labels.shape[0], keep_prob: 1.0})
			pvm = value.eval(feed_dict={input_prices: validation_data, labels: validation_labels,
				init_weights: input_weights, batch_size: validation_labels.shape[0], keep_prob: 1.0})
			portfolio_weights = weights.eval(feed_dict={input_prices: validation_data, labels: validation_labels,
				init_weights: input_weights, batch_size: validation_labels.shape[0], keep_prob: 1.0})
			prnt.print_model_results(final_pvm, pvm, portfolio_weights, path_to_model_dir, 'validation')
			if (final_pvm > best_val_value):
				saver.save(sess, path_to_best_model)

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
			prnt.print_model_results(final_pvm, pvm, portfolio_weights, path_to_model_dir, 'test')
			#print('Test trading period = %d steps and %.2f days' % (price_change.shape[0], price_change.shape[0]/48.0))

	def get_cost(self):
		return self.validation_accuracy



	"""while True:
		data, labels = get_current_window()
		print(data)
		print(labels)
		weights = weights.eval(feed_dict={input_prices: data, labels: labels, 
			init_weights: input_weights, keep_prob: 1.0})
		get_new_portfolio(weights)
		time.sleep(1800)"""

			# Proper validation test
			# validation_weights = np.zeros((1, validation_labels.shape[1]))
			# validation_weights[0,0]  = 1.0
			# v = np.ones((validation_labels.shape[0]))
			# portfolio_value = 1.0
			# for i in range(0, validation_labels.shape[0]):
			# 	v_labels = np.reshape(validation_labels[i,:], (1, validation_labels.shape[1]))
			# 	v_data = np.reshape(validation_data[i,:], (1, validation_data.shape[1], validation_data.shape[2], validation_data.shape[3]))
			# 	v[i] = final_value.eval(feed_dict={input_prices: v_data, labels: v_labels,
			# 				init_weights: validation_weights, batch_size: 1, keep_prob: 1.0})
			# 	portfolio_value = portfolio_value*v[i]
			# 	validation_weights = weights.eval(feed_dict={input_prices: v_data, labels: v_labels,
			#   		init_weights: validation_weights, batch_size: 1, keep_prob: 1.0})
			# np.savetxt('tmp/proper_validation_returns.out', v, fmt='%.8f', delimiter=' ')
			# print('The (proper) final validation set value multiplier is %.8f' % portfolio_value)

if __name__ == '__main__':
	main()
