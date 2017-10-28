from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf
import csv

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib


prices = {}
with open('data/BTC_ETH.csv', 'rb') as file:
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
with open('data/BTC_STEEM.csv') as file:
	reader = csv.reader(file, delimiter = ' ', quotechar='|')
	btc_steem = []
	for row in reader:
		btc_steem.append(row)
	prices['BTC_STEEM']=btc_steem
with open('data/BTC_BTS.csv') as file:
	reader = csv.reader(file, delimiter = ' ', quotechar='|')
	btc_bts = []
	for row in reader:
		btc_bts.append(row)
	prices['BTC_BTS']=btc_bts
with open('data/BTC_STRAT.csv') as file:
	reader = csv.reader(file, delimiter = ' ', quotechar='|')
	btc_strat = []
	for row in reader:
		btc_strat.append(row)
	prices['BTC_STRAT']=btc_strat
with open('data/BTC_XMR.csv') as file:
	reader = csv.reader(file, delimiter = ' ', quotechar='|')
	btc_xmr = []
	for row in reader:
		btc_xmr.append(row)
	prices['BTC_XMR']=btc_xmr
with open('data/BTC_ZEC.csv') as file:
	reader = csv.reader(file, delimiter = ' ', quotechar='|')
	btc_zec = []
	for row in reader:
		btc_zec.append(row)
	prices['BTC_ZEC']=btc_zec
with open('data/BTC_GNO.csv') as file:
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

window_size = 50
step_size = 40
normalized_price_array = [[]]
for coin_prices in price_array:
	while(len(coin_prices)<max_length):
		coin_prices.insert(0,0.01)









def cnn_model(input_layer):
	conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=12,
      kernel_size=[12, 4],
      padding="same",
      activation=tf.nn.relu)

	dense = tf.layers.dense(inputs=conv1, units=500, activation=tf.nn.relu)
	dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == learn.ModeKeys.TRAIN)
	out = tf.layers.dense(inputs=dropout, units=12, activation=tf.nn.softmax)

