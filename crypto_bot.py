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

# if we have already calculated the global price array, read it from txt

def read_data():
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
		with open('data/BTC_LTC.csv', 'rb') as file:
			reader = csv.reader(file, delimiter = ' ', quotechar='|')
			btc_ltc = []
			for row in reader:
				btc_ltc.append(row)
			prices['BTC_LTC']=btc_ltc
		with open('data/BTC_XRP.csv', 'rb') as file:
			reader = csv.reader(file, delimiter = ' ', quotechar='|')
			btc_xrp = []
			for row in reader:
				btc_xrp.append(row)
			prices['BTC_XRP']=btc_xrp
		with open('data/BTC_ETC.csv', 'rb') as file:
			reader = csv.reader(file, delimiter = ' ', quotechar='|')
			btc_etc = []
			for row in reader:
				btc_etc.append(row)
			prices['BTC_ETC']=btc_etc
		with open('data/BTC_XEM.csv', 'rb') as file:
			reader = csv.reader(file, delimiter = ' ', quotechar='|')
			btc_xem = []
			for row in reader:
				btc_xem.append(row)
			prices['BTC_XEM']=btc_xem
		with open('data/BTC_DASH.csv', 'rb') as file:
			reader = csv.reader(file, delimiter = ' ', quotechar='|')
			btc_dash = []
			for row in reader:
				btc_dash.append(row)
			prices['BTC_DASH']=btc_dash
		with open('data/BTC_STEEM.csv', 'rb') as file:
			reader = csv.reader(file, delimiter = ' ', quotechar='|')
			btc_steem = []
			for row in reader:
				btc_steem.append(row)
			prices['BTC_STEEM']=btc_steem
		with open('data/BTC_BTS.csv', 'rb') as file:
			reader = csv.reader(file, delimiter = ' ', quotechar='|')
			btc_bts = []
			for row in reader:
				btc_bts.append(row)
			prices['BTC_BTS']=btc_bts
		with open('data/BTC_STRAT.csv', 'rb') as file:
			reader = csv.reader(file, delimiter = ' ', quotechar='|')
			btc_strat = []
			for row in reader:
				btc_strat.append(row)
			prices['BTC_STRAT']=btc_strat
		with open('data/BTC_XMR.csv', 'rb') as file:
			reader = csv.reader(file, delimiter = ' ', quotechar='|')
			btc_xmr = []
			for row in reader:
				btc_xmr.append(row)
			prices['BTC_XMR']=btc_xmr
		with open('data/BTC_ZEC.csv','rb') as file:
			reader = csv.reader(file, delimiter = ' ', quotechar='|')
			btc_zec = []
			for row in reader:
				btc_zec.append(row)
			prices['BTC_ZEC']=btc_zec
		with open('data/BTC_GNO.csv', 'rb') as file:
			reader = csv.reader(file, delimiter = ' ', quotechar='|')
			btc_gno = []
			for row in reader:
				btc_gno.append(row)
			prices['BTC_GNO']=btc_gno

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

		global_price_array = np.array(price_array).astype('float')

		return global_price_array


window_size = 50
stride = 40


#TODO change to account for numepochs
def split_data(train_size,validation_size,test_size):
	train = global_price_array[:train_size]
	validation = global_price_array[train_size:train_size+validation_size]
	test = global_price_array[train_size+validation_size:]
	return train,validation,test



def get_local_prices(window_size,stride,global_price_array,current_step):
	start = current_step*stride
	stop = start+window_size
	if(stop<global_price_array.shape()[1]):
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
	length = array.shape()[1]
	n = length/stride - 1
	train = []
	price_changes = []
	for i in range(n-1):
		prices = get_local_prices(window_size,stride,array,i)
		train.append(prices[0])
		price_changes.append(prices[1])
	return np.array(train),np.array(price_changes)






def cnn_model_fn(input_layer,mode,numFilters,filterSize,hiddenUnits):
	conv1 = tf.layers.conv2d(
			inputs=input_layer,
			filters=self.numFilters,
			kernel_size=self.filterSize,
			padding="same",
			activation=tf.nn.relu)
	dense = tf.layers.dense(inputs=conv1, units=self.hiddenUnits, activation=tf.nn.relu)
	dropout = tf.layers.dropout(
		inputs=dense, rate=0.4, training= (mode == learn.modeKeys.TRAIN))
	out = tf.layers.dense(inputs=dropout, units=12, activation=tf.nn.softmax)
	loss = tf.losses.log_loss(out,price_change)

	if mode == tf.estimator.ModeKeys.TRAIN:
		optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00001)
		train_op = optimizer.minimize(
	        loss=loss,
	        global_step=tf.train.get_global_step())
		return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
	if mode == tf.estimator.ModeKeys.PREDICT:
		return out



"""mode = 'train'
local = get_local_prices(window_size,step_size,global_price_array,0)
local = tf.convert_to_tensor(local,'float32')
local = tf.reshape(local,[1,12,50,1])
out = cnn_model(local)"""

def main():
	window_size = 50
	stride = 40
	global_price_array = read_data()
	total_time_steps = len(global_price_array[0])
	train_size = int(total_time_steps*0.7)
	validation_size = int(total_time_steps*0.15)
	test_size = int(total_time_steps*0.15)
	train, validation, test = split_data(global_price_array,train_size,validation_size,test_size)
	train_data, train_labels = get_data(train,window_size,stride)
	test_data, test_labels = get_data(test,window_size,stride)
	test_data, test_labels = get_data()
	classifier = tf.estimator.Estimator(
		model_fn = cnn_model_fn, model_dir = "/tmp/cnn_model"
	)

	train_input_fn = tf.estimator.inputs.numpy_input_fn(
		x = {"x":train_data},
		y = train_labels,
		batch_size = 50,
		num_epochs = None,
		shuffle=True)
	classifier.train(
		input_fn=train_input_fn,
		steps = 900000,
		)

	eval_input_fn = tf.estimator.inputs.numpy_input_fn(
		x={"x": test_data},
		y=test_labels,
		num_epochs=1,
		shuffle=False)
	eval_results = classifier.evaluate(input_fn=eval_input_fn)
	print(eval_results)










