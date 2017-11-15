from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf
import csv

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

global_price_array = []
def read_data():
	# if we have already calculated the global price array, read it from txt
	try:
		global_price_array = np.genfromtxt('global_price')
	except:
		prices = {}
		with open('data/BTC_ETH.csv', 'r') as file:
			reader = csv.reader(file, delimiter=' ', quotechar='|')
			btc_eth = []
			for row in reader:
				btc_eth.append(row)
			prices['BTC_ETH']=btc_eth
		with open('data/BTC_LTC.csv', 'r') as file:
			reader = csv.reader(file, delimiter = ' ', quotechar='|')
			btc_ltc = []
			for row in reader:
				btc_ltc.append(row)
			prices['BTC_LTC']=btc_ltc
		with open('data/BTC_XRP.csv', 'r') as file:
			reader = csv.reader(file, delimiter = ' ', quotechar='|')
			btc_xrp = []
			for row in reader:
				btc_xrp.append(row)
			prices['BTC_XRP']=btc_xrp
		with open('data/BTC_ETC.csv', 'r') as file:
			reader = csv.reader(file, delimiter = ' ', quotechar='|')
			btc_etc = []
			for row in reader:
				btc_etc.append(row)
			prices['BTC_ETC']=btc_etc
		with open('data/BTC_XEM.csv', 'r') as file:
			reader = csv.reader(file, delimiter = ' ', quotechar='|')
			btc_xem = []
			for row in reader:
				btc_xem.append(row)
			prices['BTC_XEM']=btc_xem
		with open('data/BTC_DASH.csv', 'r') as file:
			reader = csv.reader(file, delimiter = ' ', quotechar='|')
			btc_dash = []
			for row in reader:
				btc_dash.append(row)
			prices['BTC_DASH']=btc_dash
		with open('data/BTC_STEEM.csv', 'r') as file:
			reader = csv.reader(file, delimiter = ' ', quotechar='|')
			btc_steem = []
			for row in reader:
				btc_steem.append(row)
			prices['BTC_STEEM']=btc_steem
		with open('data/BTC_BTS.csv', 'r') as file:
			reader = csv.reader(file, delimiter = ' ', quotechar='|')
			btc_bts = []
			for row in reader:
				btc_bts.append(row)
			prices['BTC_BTS']=btc_bts
		with open('data/BTC_STRAT.csv', 'r') as file:
			reader = csv.reader(file, delimiter = ' ', quotechar='|')
			btc_strat = []
			for row in reader:
				btc_strat.append(row)
			prices['BTC_STRAT']=btc_strat
		with open('data/BTC_XMR.csv', 'r') as file:
			reader = csv.reader(file, delimiter = ' ', quotechar='|')
			btc_xmr = []
			for row in reader:
				btc_xmr.append(row)
			prices['BTC_XMR']=btc_xmr
		with open('data/BTC_ZEC.csv','r') as file:
			reader = csv.reader(file, delimiter = ' ', quotechar='|')
			btc_zec = []
			for row in reader:
				btc_zec.append(row)
			prices['BTC_ZEC']=btc_zec
		# Commented out by John - only 12 currencies (including btc_btc) to start
		# with open('data/BTC_GNO.csv', 'r') as file:
		# 	reader = csv.reader(file, delimiter = ' ', quotechar='|')
		# 	btc_gno = []
		# 	for row in reader:
		# 		btc_gno.append(row)
		# 	prices['BTC_GNO']=btc_gno

		dates = []
		for key in prices.keys():
			dates.append( (prices[key][1][0].split(',')[0], key)  )

		price_array = [[]]
		for val in prices.values():
			price_array.append([row[0].split(',')[1] for row in val][1:])
		price_array = price_array[1:]
		max_length = len(max(price_array, key = lambda x: len(x)))

		for coin_prices in price_array:
			while(len(coin_prices)<max_length):
				coin_prices.insert(0,"NaN")

		global_price_array = np.array(price_array, dtype=np.float16)

		# Insert row of ones for btc_btc ratio
		btc_btc = np.ones( (1,max_length), dtype=np.float16)
		global_price_array = np.insert(global_price_array, 0, btc_btc, axis=0)
		
		return global_price_array

#TODO change to account for numepochs
def split_data(global_price_array,train_size,validation_size,test_size):
	train = global_price_array[:train_size]
	validation = global_price_array[train_size:train_size+validation_size]
	test = global_price_array[train_size+validation_size:]
	return train,validation,test

def get_local_prices(window_size,stride,global_price_array,current_step):
	start = current_step*stride
	stop = start+window_size
	if(stop<global_price_array.shape[1]):
		local = global_price_array[:,start:stop]
		last = local[:,(window_size-1):window_size]
		normalized = np.divide(local,last)
		normalized[np.isnan(normalized)==True]=0.01
		shift = global_price_array[:,start+1:stop+1]
		a = shift[:,(window_size-1):window_size]
		normalized_shift =np.divide(shift,a)
		normalized_shift[np.isnan(normalized_shift)==True]=0.01
		price_change = np.divide(normalized_shift,normalized)
		return normalized,price_change

def get_data(array,window_size,stride):
	length = array.shape[1]
	n = int(length/stride) - 1
	train = []
	price_changes = []
	for i in range(n-1):
		prices = get_local_prices(window_size,stride,array,i)
		train.append(prices[0])
		price_changes.append(prices[1][:,window_size-1:window_size])
	return np.array(train),np.array(price_changes)

def cnn_model_fn(features, labels, mode):
	"""Model function for a CNN."""
	input_layer = tf.reshape(features["x"], [-1, num_filters, window_size, 1])
	labels = tf.reshape(labels, [-1, numCoins])
	conv1 = tf.layers.conv2d(
			inputs=input_layer,
			filters=num_filters,
			kernel_size=filterSize,
			strides=(1,1),
			padding="valid",
			activation=tf.nn.relu)
	conv1_flat = tf.reshape(conv1, [60, num_filters * 47])
	dense = tf.layers.dense(inputs=conv1_flat, units=hiddenUnits, activation=tf.nn.relu)
	dropout = tf.layers.dropout(
		inputs=dense, rate=0.4, training= (mode == learn.ModeKeys.TRAIN))
	out = tf.layers.dense(inputs=dropout, units=numCoins, activation=tf.nn.softmax)
	predictions = {
		"best_guess": tf.argmax(input=out, axis=1),
		"weights": tf.identity(out, name="porfolio_weights")
	}
	loss = tf.losses.log_loss(out, labels)
	if mode == tf.estimator.ModeKeys.TRAIN:
		optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
		train_op = optimizer.minimize(
			loss=loss,
			global_step=tf.train.get_global_step())
		return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
		# return out

# Hyperparameters
numCoins = 12
window_size = 50
stride = 40
filterSize = [numCoins, 4]
hiddenUnits = 500
num_filters = 12

tf.logging.set_verbosity(tf.logging.INFO)

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










