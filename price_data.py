from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import csv
import pdb

global_price_array = []
def read_data():
	# if we have already calculated the global price array, read it from txt
	try:
		raise Exception('err')
		global_price_array = np.genfromtxt('global_price')
	except:
		prices = {}
		# with open('data/BTC_ETH.csv', 'r') as file:
		with open('data/test/btc_eth_test.csv', 'r') as file:
			reader = csv.reader(file, delimiter=' ', quotechar='|')
			btc_eth = []
			for row in reader:
				btc_eth.append(row)
			prices['BTC_ETH']=btc_eth
		# with open('data/BTC_LTC.csv', 'r') as file:
		with open('data/test/btc_ltc_test.csv', 'r') as file:
			reader = csv.reader(file, delimiter = ' ', quotechar='|')
			btc_ltc = []
			for row in reader:
				btc_ltc.append(row)
			prices['BTC_LTC']=btc_ltc
		# with open('data/BTC_XRP.csv', 'r') as file:
		with open('data/test/btc_xrp_test.csv', 'r') as file:
			reader = csv.reader(file, delimiter = ' ', quotechar='|')
			btc_xrp = []
			for row in reader:
				btc_xrp.append(row)
			prices['BTC_XRP']=btc_xrp
		# with open('data/BTC_ETC.csv', 'r') as file:
		with open('data/test/btc_etc_test.csv', 'r') as file:
			reader = csv.reader(file, delimiter = ' ', quotechar='|')
			btc_etc = []
			for row in reader:
				btc_etc.append(row)
			prices['BTC_ETC']=btc_etc
		# with open('data/BTC_XEM.csv', 'r') as file:
		with open('data/test/btc_xem_test.csv', 'r') as file:
			reader = csv.reader(file, delimiter = ' ', quotechar='|')
			btc_xem = []
			for row in reader:
				btc_xem.append(row)
			prices['BTC_XEM']=btc_xem
		# with open('data/BTC_DASH.csv', 'r') as file:
		with open('data/test/btc_dash_test.csv', 'r') as file:
			reader = csv.reader(file, delimiter = ' ', quotechar='|')
			btc_dash = []
			for row in reader:
				btc_dash.append(row)
			prices['BTC_DASH']=btc_dash
		# with open('data/BTC_STEEM.csv', 'r') as file:
		with open('data/test/btc_steem_test.csv', 'r') as file:
			reader = csv.reader(file, delimiter = ' ', quotechar='|')
			btc_steem = []
			for row in reader:
				btc_steem.append(row)
			prices['BTC_STEEM']=btc_steem
		# with open('data/BTC_BTS.csv', 'r') as file:
		with open('data/test/btc_bts_test.csv', 'r') as file:
			reader = csv.reader(file, delimiter = ' ', quotechar='|')
			btc_bts = []
			for row in reader:
				btc_bts.append(row)
			prices['BTC_BTS']=btc_bts
		# with open('data/BTC_STRAT.csv', 'r') as file:
		with open('data/test/btc_strat_test.csv', 'r') as file:
			reader = csv.reader(file, delimiter = ' ', quotechar='|')
			btc_strat = []
			for row in reader:
				btc_strat.append(row)
			prices['BTC_STRAT']=btc_strat
		# with open('data/BTC_XMR.csv', 'r') as file:
		with open('data/test/btc_xmr_test.csv', 'r') as file:
			reader = csv.reader(file, delimiter = ' ', quotechar='|')
			btc_xmr = []
			for row in reader:
				btc_xmr.append(row)
			prices['BTC_XMR']=btc_xmr
		# with open('data/BTC_ZEC.csv','r') as file:
		with open('data/test/btc_zec_test.csv', 'r') as file:
			reader = csv.reader(file, delimiter = ' ', quotechar='|')
			btc_zec = []
			for row in reader:
				btc_zec.append(row)
			prices['BTC_ZEC']=btc_zec

		dates = []
		for key in prices.keys():
			dates.append( (prices[key][1][0].split(',')[0], key)  )

		high_price_array = [[]]
		open_price_array = [[]]
		volume_array = [[]]
		for val in prices.values():
			high_price_array.append([row[0].split(',')[1] for row in val][1:])
			open_price_array.append([row[0].split(',')[3] for row in val][1:])
			volume_array.append([row[0].split(',')[5] for row in val][1:])
		high_price_array = high_price_array[1:]
		open_price_array = open_price_array[1:]
		volume_array = volume_array[1:]
		max_length = len(max(high_price_array, key = lambda x: len(x)))

		for coin_prices in high_price_array:
			while(len(coin_prices)<max_length):
				coin_prices.insert(0, coin_prices[0])
		for coin_prices in open_price_array:
			while(len(coin_prices)<max_length):
				coin_prices.insert(0, coin_prices[0])
		for coin_volumes in volume_array:
			while(len(coin_volumes)<max_length):
				coin_volumes.insert(0, coin_volumes[0])

		global_volume_array = np.array(volume_array, dtype=np.float32)
		global_high_price_array = np.array(high_price_array, dtype=np.float32)
		global_open_price_array = np.array(open_price_array, dtype=np.float32)
		global_dp_dt_array = calc_dp_dt_array(global_high_price_array, 1.0)


		input_array = np.stack([global_high_price_array,global_open_price_array,global_volume_array,global_dp_dt_array],axis=2)

		return input_array


def calc_dp_dt_array(p, h):
	dp_dt_array = np.zeros(p.shape, dtype=np.float32)
	# loop over num_coins
	for i in range(p.shape[0]): 
		# loop over all time steps
		for j in range(p.shape[1]): 
			if j == 0: 
				# 1st order forward finite difference method
				dp_dt_array[i,j] = p[i,j+1]-p[i,j]
			elif j == p.shape[1]-1: 
				# 1st order backward finite difference method
				dp_dt_array[i,j] = p[i,j]-p[i,j-1]
			elif (j == 1 or j == p.shape[1]-2): 
				# 2nd order finite difference method
				dp_dt_array[i,j] = (p[i,j+1]-p[i,j-1]) / 2.0
			else: 
				# a 4th order finite difference method
				dp_dt_array[i,j] = (8.0*(p[i,j+1]-p[i,j-1]) - (p[i,j+2]-p[i,j-2]) / 12.0)
	dp_dt_array = np.divide(dp_dt_array, h)
	return dp_dt_array


#TODO change to account for numepochs
def split_data(global_price_array,train_size,validation_size,test_size):
	train = global_price_array[:,:train_size]
	validation = global_price_array[:,train_size:train_size+validation_size]
	test = global_price_array[:,train_size+validation_size:]
	return train,validation,test

def get_local_prices(window_size,stride,global_price_array,current_step):
	start = current_step*stride
	stop = start+window_size
	if(stop<global_price_array.shape[1]):
		local = global_price_array[:,start:stop]
		last = local[:,(window_size-1):window_size]
		normalized = np.divide(local,np.abs(last))
		normalized[np.isnan(normalized)==True]=0.01
		normalized[np.isinf(normalized)==True]=0.01
		shift = global_price_array[:,start+1:stop+1]
		a = shift[:,(window_size-1):window_size]
		normalized_shift = np.divide(shift,a)
		normalized_shift[np.isnan(normalized_shift)==True]=0.01
		normalized_shift[np.isinf(normalized_shift)==True]=0.01
		# price_change = np.divide(normalized_shift,normalized)
		price_change = np.divide(shift, local)
		price_change[np.isnan(price_change)==True]=0.01
		price_change[np.isinf(price_change)==True]=0.01
		price_change = price_change[:,:,0]#remove volume and open
		#pdb.set_trace()
		return normalized,price_change

def get_data(array,window_size,stride):
	length = array.shape[1]
	n = int(length/stride) - 1
	train = []
	price_changes = []
	for i in range(n-window_size):
		prices, pc = get_local_prices(window_size,stride,array,i)
		train.append(prices)
		price_changes.append(pc[:,window_size-1:window_size])
	#pdb.set_trace()
	#print(np.array(price_changes))
	return np.array(train),np.array(price_changes)

def get_next_price_batch(prices, price_changes, batch_size, num_coins, training_step):
	start_index = training_step % (prices.shape[0]-batch_size)
	p = prices[start_index:start_index+batch_size,:,:]
	p_c = price_changes[start_index:start_index+batch_size,:,:]
	p_c = np.reshape(p_c, (batch_size, num_coins))
	btc_btc = np.ones( (1, batch_size), dtype=np.float32)
	p_c = np.insert(p_c, 0, btc_btc, axis=1)
	#pdb.set_trace()
	return p, p_c







