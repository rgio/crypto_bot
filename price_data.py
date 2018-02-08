from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import csv
import ipdb
import pdb
import time
from poloniex_api import *

global_price_array = []
def read_data(coin_list, directory='data/test/'):
	# if we have already calculated the global price array, read it from txt
	try:
		raise Exception('err')
		global_price_array = np.genfromtxt('global_price')
	except:
		prices = {}
        	for coin in coin_list:
			with open(directory+'btc_{}_test.csv'.format(coin), 'r') as file:
				reader = csv.reader(file, delimiter=' ', quotechar='|')
				btc_coin = []
				for row in reader:
					btc_coin.append(row)
				prices['BTC_{}'.format(coin.upper())]=btc_coin

		dates = []
		for key in prices.keys():
			dates.append( (prices[key][1][0].split(',')[0], key)  )

		high_price_array = [[]]
		low_price_array = [[]]
		open_price_array = [[]]
		volume_array = [[]]
		for key in prices.keys():
			high_price_array.append([row[0].split(',')[1] for row in prices[key]][1:])
			low_price_array.append([row[0].split(',')[2] for row in prices[key]][1:])
			open_price_array.append([row[0].split(',')[3] for row in prices[key]][1:])
			volume_array.append([row[0].split(',')[5] for row in prices[key]][1:])
		high_price_array = high_price_array[1:]
		low_price_array = low_price_array[1:]
		open_price_array = open_price_array[1:]
		volume_array = volume_array[1:]
		max_length = len(max(high_price_array, key = lambda x: len(x)))

		for coin_prices in high_price_array:
			while(len(coin_prices)<max_length):
				coin_prices.insert(0, coin_prices[0])
		for coin_prices in low_price_array:
			while(len(coin_prices)<max_length):
				coin_prices.insert(0, coin_prices[0])
		for coin_prices in open_price_array:
			while(len(coin_prices)<max_length):
				coin_prices.insert(0, coin_prices[0])
		for coin_volumes in volume_array:
			while(len(coin_volumes)<max_length):
				coin_volumes.insert(0, coin_volumes[0])

		#pdb.set_trace()
		global_volume_array = np.array(volume_array, dtype=np.float32)
		global_high_price_array = np.array(high_price_array, dtype=np.float32)
		global_low_price_array = np.array(low_price_array, dtype=np.float32)
		global_open_price_array = np.array(open_price_array, dtype=np.float32)
		global_dp_dt_array = calc_dp_dt_array(global_high_price_array, 1.0)


		input_array = np.stack([global_high_price_array,global_low_price_array,global_open_price_array,global_volume_array,global_dp_dt_array],axis=2)
		#pdb.set_trace()

		sampled_array = np.zeros( (input_array.shape[0],int(input_array.shape[1]/6+1),input_array.shape[2]) )
		for i in range(input_array.shape[1]):
			if i%6==0:
				local = input_array[:,i:i+6,:]
				lows = np.array([min(row) for row in local[:,:,1]])
				highs = np.array([max(row) for row in local[:,:,0]])
				volume = np.array([sum(row) for row in local[:,:,3]])
				sampled_array[:,int(i/6),0]= highs
				sampled_array[:,int(i/6),1]= lows
				sampled_array[:,int(i/6),2]= input_array[:,i,2]
				sampled_array[:,int(i/6),3]= volume
				sampled_array[:,int(i/6),4]= input_array[:,i,4]
				#pdb.set_trace()
				#sampled_array[:,int(i/6),:]=input_array[:,i,:]
		#pdb.set_trace()
		#return input_array
		return sampled_array


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
	#pdb.set_trace()
	dp_dt_array = np.divide(dp_dt_array, h)
	return dp_dt_array

def get_current_window():
	t = time.time()
	t0 = t-(100000)
	fetch_data(start_time=t0)
	array = read_data('live_data/')
	window = array[:,-50,:]
	#pdb.set_trace()
	return window


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

		"""high_prices = normalized[:,:,0]
		dp_dt_array = np.zeros(high_prices.shape)
		for j in range(high_prices.shape[1]):
			if j == 0:
				dp_dt_array[:,j] = high_prices[:,j+1]-high_prices[:,j]
			elif j == high_prices.shape[1]-1:
				dp_dt_array[:,j] = high_prices[:,j]-high_prices[:,j-1]
			elif (j == 1 or j == high_prices.shape[1]-2):
				dp_dt_array[:,j] = (high_prices[:,j+1]-high_prices[:,j-1])/2.0
			else:
				dp_dt_array[:,j] = (8.0*(high_prices[:,j+1]-high_prices[:,j-1]) - (high_prices[:,j+2]-high_prices[:,j-2])/12.0 )
		last = np.reshape(dp_dt_array[:,high_prices.shape[1]-1],(high_prices.shape[0],1))
		dp_dt_normalized = np.divide(dp_dt_array,np.abs(last))
		dp_dt_normalized[np.isnan(dp_dt_normalized)==True]=0.01
		dp_dt_normalized[np.isinf(dp_dt_normalized)==True]=0.01

		normalized = np.stack([normalized[:,:,0],normalized[:,:,1],normalized[:,:,2],dp_dt_normalized],axis=2)"""

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
	return p, p_c, start_index


