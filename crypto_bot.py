from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports of general code
import numpy as np
import tensorflow as tf

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

# Imports of our own code
from price_data import *
from cnn import *

# Set up logging
tf.logging.set_verbosity(tf.logging.INFO)

# Model hyperparameters
window_size = 50
stride = 40
transaction_cost = 0.0025 # 0.25% commission fee for each transaction

def main():
	# Load training and eval data
	global_price_array = read_data()
	total_time_steps = len(global_price_array[0])
	train_size = int(total_time_steps*0.7)
	validation_size = int(total_time_steps*0.15)
	test_size = int(total_time_steps*0.15)
	train, validation, test = split_data(global_price_array,train_size,validation_size,test_size)
	train_data, train_labels = get_data(train,window_size,stride)
	test_data, test_labels = get_data(test,window_size,stride)
	# Create the estimator
	classifier = tf.estimator.Estimator(
		model_fn = cnn_model_fn, model_dir = "/tmp/cnn_model"
	)
	# Set up logging
	tensors_to_log = {"weights": "porfolio_weights"}
	logging_hook = tf.train.LoggingTensorHook(
	 	tensors=tensors_to_log, every_n_iter=100)
	# Train the model
	print('Begin training...')
	train_input_fn = tf.estimator.inputs.numpy_input_fn(
		x = {"x": train_data},
		y = train_labels,
		batch_size = 60,
		num_epochs = None,
		shuffle=True)
	classifier.train(
		input_fn=train_input_fn,
		steps = 900000,
		hooks=[logging_hook])
	# Evaluate the model and print results
	print('Begin evaluation...')
	eval_input_fn = tf.estimator.inputs.numpy_input_fn(
		x={"x": test_data},
		y=test_labels,
		num_epochs=1,
		shuffle=False)
	eval_results = classifier.evaluate(input_fn=eval_input_fn)
	print(eval_results)





