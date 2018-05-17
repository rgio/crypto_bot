from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import csv
import pdb
import time
import poloniex_api as pnx
from sklearn.preprocessing import Imputer

global_price_array = []
DEFAULT_DIRECTORY = 'data/'
PAIRS = ["BTC_BTS","BTC_ZEC","BTC_STRAT","BTC_XEM","BTC_STEEM","BTC_LTC","BTC_ETC","BTC_XRP","BTC_XMR","BTC_DASH","BTC_ETH",
   "BTC_STR", "BTC_LSK", "BTC_DOGE", "BTC_SC", "BTC_SYS", "BTC_DGB", "BTC_MAID", "BTC_NXT", "BTC_BCN"] # 21 total coins

import poloniex_api as pnx

def get_price_data(test=False):
		try:
			if test:
				input_array = read_data('data/test/')
			else:
				input_array = read_data()
		
		except:
			if test:
				pnx.fetch_data(test=test)
				input_array = pread_data('data/test/')
			else:
				pnx.fetch_data()
				input_array = read_data()

		return input_array


def read_data(directory=DEFAULT_DIRECTORY, pairs=PAIRS):
	# if we have already calculated the global price array, read it from txt
	# TODO: The second line of the try statement can never be executed and the except clause is executed no matter what.
	# TODO: Maybe we can remove 'raise Exception('err')'? This syntax is confusing and exception handling can be expensive
	try:
		raise Exception('err')
		global_price_array = np.genfromtxt('global_price')
	except:
		prices = {}
		for pair in pairs:
			with open(directory+pair+"_test.csv") as file:
				reader = csv.reader(file, delimiter=' ', quotechar='|')
				pair_info = []
				for row in reader:
					pair_info.append(row)
				prices[pair]=pair_info

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
				sampled_array[:,int(i/6),:]=input_array[:,i,:]
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
				# 2nd order centered finite difference method
				dp_dt_array[i,j] = (p[i,j+1]-p[i,j-1]) / 2.0
			else:
				# a 4th order finite difference method
				dp_dt_array[i,j] = (8.0*(p[i,j+1]-p[i,j-1]) - (p[i,j+2]-p[i,j-2]) / 12.0)
	dp_dt_array = np.divide(dp_dt_array, h)
	return dp_dt_array

def calc_uniform_portfolio(price_change):
	uniform_portfolio = np.full(price_change.shape[1], 1.0/price_change.shape[1])
	# TODO: calculate portfolio return for an initial uniform portfolio with
	# 		no redistribution throughout the entire period
	unchanged_uniform_portfolio = np.full(price_change.shape[1], 1.0/price_change.shape[1])
	uniform_portfolio_value = 1.0
	for i in range(0, price_change.shape[0]):
		multiplier = 0.0
		for j in range(0, price_change.shape[1]):
			# TODO: add in transaction cost for reweighting to uniform portfolio
			multiplier += price_change[i,j] * uniform_portfolio[j]
		# 	unchanged_multiplier += price_change[i,j] * unchanged_uniform_portfolio[j]
		# 	unchanged_uniform_portfolio[j] =
		# unchanged_uniform_portfolio
		uniform_portfolio_value *= multiplier
	print('The uniform portfolio value with no transaction costs = ', uniform_portfolio_value)
	return uniform_portfolio_value

def calc_optimal_portfolio(price_change, save_path):
	# optimal portfolio has entire portfolio in the asset with largest price change
	opt_porfolio_index = np.argmax(price_change, axis=1)
	opt_portfolio_value = 1.0
	optimal_portfolio = np.zeros(price_change.shape)
	optimal_return = np.zeros(opt_porfolio_index.shape)
	for i in range(0,price_change.shape[0]):
		for j in range(0, price_change.shape[1]):
			if (j == opt_porfolio_index[i]):
				optimal_return[i] = price_change[i,j]
				optimal_portfolio[i,j] = 1.0
				opt_portfolio_value *= price_change[i,j]
			else:
				optimal_portfolio[i,j] = 0.0
	np.savetxt(save_path + 'labels.out', price_change, fmt='%.8f', delimiter=' ')
	np.savetxt(save_path + 'optimal_portfolio_index.out', opt_porfolio_index, fmt='%d')
	np.savetxt(save_path + 'optimal_portfolio_return.out', optimal_return, fmt='%.8f')
	np.savetxt(save_path + 'optimal_portfolio.out', optimal_portfolio, fmt='%.1f', delimiter=' ')
	print('')
	print('Directory = %s trading period = %d steps and %.2f days' % (save_path, price_change.shape[0], price_change.shape[0]/48.0))
	print('The optimal portfolio value with no transaction costs = ', opt_portfolio_value)
	calc_uniform_portfolio(price_change)
	return optimal_portfolio, opt_portfolio_value

def get_current_window():
	t = time.time()
	t0 = t-(100000)
	pnx.fetch_data(start_time=t0, path='live_data/')
	array = read_data('live_data/')
	array = array.astype(np.float32)
	data = array[:,-51:-1,:]
	data = np.reshape(data, (1, data.shape[0], data.shape[1], data.shape[2]))
	labels = array[:,-1:,0]
	labels = np.insert(labels,0,1,axis=0)#BTC
	labels = labels.T
	return data, labels

def split_data(global_price_array, train_size, val_size, test_size):
	# TODO: Is this the best way to split up the data?
	total_time_steps = global_price_array.shape[1]
	train_len = int(total_time_steps * train_size)
	val_len = int(total_time_steps * val_size)
	test_len = int(total_time_steps * test_size)

	train = global_price_array[:, :train_len]
	val = global_price_array[:, train_len:train_len + val_len]
	test = global_price_array[:, train_len + val_len:]
	return train, val, test

def nan_helper(array):
    """Helper to handle indices and logical indices of NaNs.

    args:
        array (np.ndarray): array with possible np.nan values
    
    Output:
        nans (int): logical indices of NaNs
        index (function): a function, with signature indices= index(logical_indices),
        	to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]

def impute_missing_and_inf(array, default=1.0):
	"""
	Imputes missing or infinite values in array by trying to average the
	neighboring values, and if that doesn't work, setting it to the default 
	value.
	
	args:
		array (np.ndarray): array to have missing values imputed
		default (float): default value if neighboring values cannot be used for
			impuation
	
	returns:
		imputed_array (np.ndarray): array with missing and infinite values
			imputed
	"""
	nans = np.isnan(array)
	indexer = lambda z: z.nonzero()[0]
	array[nans] = np.interp(indexer(nans), indexer(~nans), array[~nans])
	return array

def get_local_prices(global_price_array, window_size, stride, current_step):
	# TODO: add 5 metrics to documentation
	# TODO: do we need to compute 
	"""
	Gets normalized price window and relative price change array for a given
	stride step in the global_price_array.

	args:
		global_price_array (np.ndarray): m x n x p array of p metrics of m
			currencies over n timesteps. Metrics are WHAT ARE METRICS
		window_size (int): number of timesteps to include in each window
		stride (int): distance between starting indices of consecutive windows
	"""
	start = current_step * stride
	stop = start + window_size
	end = global_price_array.shape[1]

	if stop < end:
		local_window = global_price_array[:,start:stop]
		local_window_shift = global_price_array[:,start+1:stop+1]

		last_val = local_window[:,(window_size-1):window_size]
		#lost_val_shift = local_window_shift[:,-1]

		# TODO: could try norm based on mean or max or sum=1
		norm_local_window = np.divide(local_window, np.abs(last_val))
		price_change_array = np.divide(local_window, local_window_shift)
		# remove volume and open
		price_change_array = price_change_array[:,:,0]

		norm_local_window = impute_missing_and_inf(norm_local_window)
		price_change_array = impute_missing_and_inf(price_change_array)

		return norm_local_window, price_change_array

def get_data(array,window_size,stride):
	num_time_steps = array.shape[1]
	n = int(num_time_steps / stride) - 1
	train = []
	price_changes = []

	for i in range(n - window_size):
		prices, pc = get_local_prices(array, window_size, stride, i)
		train.append(prices)
		price_changes.append(pc[:,window_size-1:window_size])

	return np.array(train),np.array(price_changes)

def get_random_batch_index_geometric(max_index, hparams):
	# Stochastic sampling of geometric decay, f(k) = (p)(1-p)**(k-1)
	p = hparams.geometric_decay / max_index  # Make p larger to favor more recent data
	k = np.random.geometric(p)
	while k > max_index-1:
		k = np.random.geometric(p)
	start_index = max_index - k
	return start_index

def get_random_batch_index_uniform(max_index):
	# Stochastic sampling of uniform distribution
	start_index = np.random.random_integers(1, max_index-1)
	return start_index

def get_specific_price_batch(prices, price_changes, start_index, hparams, params):
	p = prices[start_index:start_index+hparams.batch_size,:,:]
	p_c = price_changes[start_index:start_index+hparams.batch_size,:,:]
	p_c = np.reshape(p_c, (hparams.batch_size, params.num_coins))
	btc_btc = np.ones( (1, hparams.batch_size), dtype=np.float32)
	p_c = np.insert(p_c, 0, btc_btc, axis=1)
	#pdb.set_trace()
	return p, p_c, start_index

def get_next_price_batch(prices, price_changes, training_step, hparams, params):
	if hparams.batch_sampling_method == 'random_geometric':
		start_index = get_random_batch_index_geometric(prices.shape[0]-hparams.batch_size, hparams)
	elif hparams.batch_sampling_method == 'random_uniform':
		start_index = get_random_batch_index_uniform(prices.shape[0]-hparams.batch_size)
	elif hparams.batch_sampling_method == 'systematic_uniform':
		start_index = training_step % (prices.shape[0]-hparams.batch_size-1) + 1
	p, p_c, start_index = get_specific_price_batch(prices, price_changes, start_index, hparams, params)
	return p, p_c, start_index
