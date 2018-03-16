from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect

from tensorflow.contrib.training import HParams
from skopt import gp_minimize, forest_minimize
import skopt.space as space
from skopt.utils import use_named_args
from skopt.plots import plot_convergence

# local imports
import crypto_bot

################################################### SPECIFY TESTING ###################################################
																													  #
TEST = True																											  #
																													  #
#######################################################################################################################


def main():
	optimizer = HyperparameterOptimizer(test=TEST)
	results = optimizer.optimize()
	plot_convergence(results)
	print('Hyperparameeter optimization log:\n{optimizer.hparam_log}')
	print('Best hyperparameters:\n{optimizer.best_hparams}')


def set_hparams(test=False) -> HParams:
	hparams = HParams(
		batch_sampling_method='random_geometric',   # options: random_geometric, random_uniform, systematic_uniform in price_data.py
		window_size=50,
		stride=1,
		batch_size=100,
		num_training_steps=200000,
		learning_rate=2e-4,
		geometric_decay=0.5,  # the large geometric_decay is the more recent times will be selected in training
		conv_layers_separable=True,
		len_conv1_filters=3,
		num_conv1_features=8,
		num_conv2_features=32,
		num_fc1_neurons=12,   # only for option two_fc_layers; it is set to num_coins for one_fc_layer in cnn.py
		model_ending='one_fc_layer',   # options: two_fc_layers, one_fc_layer, third_conv_layer
		dropout_keep_prob=0.5,
	)
	if test:
		hparams.set_hparam('batch_size', 20)
		hparams.set_hparam('num_training_steps', 3)
	return hparams


def set_params() -> HParams:
	params = HParams(coin_pairs = ["BTC_BTS", "BTC_ZEC", "BTC_STRAT", "BTC_XEM", "BTC_STEEM", "BTC_LTC", "BTC_ETC",
								   "BTC_XRP", "BTC_XMR", "BTC_DASH", "BTC_ETH", "BTC_STR", "BTC_LSK", "BTC_DOGE",
								   "BTC_SC", "BTC_SYS", "BTC_DGB", "BTC_MAID", "BTC_NXT", "BTC_BCN"],
					 num_input_channels=4,
					 len_conv3_filters=1,)
	params.add_hparam("num_coins", len(params.coin_pairs))
	return params


def init_search_space_dict(test=False) -> dict:
	search_space_dict = dict(
		dim_batch_sampling_method=space.Categorical(categories=['random_geometric', 'random_uniform', 'systematic_uniform'],
													name='batch_sampling_method'),
		dim_window_size=space.Integer(low=10, high=1000, name='window_size'),
		dim_stride=space.Integer(low=1, high=10, name='stride'),
		dim_batch_size=space.Integer(low=10, high=1000, name='batch_size'),
		dim_num_training_steps=space.Integer(low=10000, high=5000000, name='num_training_steps'),
		dim_learning_rate=space.Real(low=1e-6, high=1e-2, prior='log-uniform', name='learning_rate'),
		dim_geometric_decay=space.Real(low=1e-6, high=1, prior='log-uniform', name='geometric_decay'),
		dim_conv_layers_seperable=space.Categorical(categories=[True, False], name='conv_layers_separable'),
		dim_len_conv1_filters=space.Integer(low=2, high=12, name='len_conv1_filters'),
		dim_num_conv1_features=space.Integer(low=1, high=64, name='num_conv1_features'),
		dim_num_conv2_features=space.Integer(low=8, high=128, name='num_conv2_features'),
		dim_num_fc1_neurons=space.Integer(low=8, high=32, name='num_fc1_neurons'),
		### Third conv layer doesn't work yet
		dim_model_ending=space.Categorical(categories=['one_fc_layer', 'two_fc_layers', 'third_conv_layer'], name='model_ending'),
		dim_dropout_keep_prob=space.Real(low=.1, high=.9, name='dropout_keep_prob'),
	)
	if test:
		search_space_dict.update({'dim_batch_size': space.Integer(low=10, high=30, name='batch_size'),
								  'dim_num_training_steps': space.Integer(low=2, high=4, name='num_training_steps')})
	return search_space_dict


def init_search_space_list(test=False) -> list:
	search_space_list = list(init_search_space_dict(test).values())
	return search_space_list


@use_named_args(dimensions=init_search_space_list(TEST))
def run_bot(batch_sampling_method,
			window_size,
			stride,
			batch_size,
			num_training_steps,
			learning_rate,
			geometric_decay,
			conv_layers_separable,
			len_conv1_filters,
			num_conv1_features,
			num_conv2_features,
			num_fc1_neurons,
			model_ending,
			dropout_keep_prob) -> float:
	# TODO: remove hardcoding of args (metaclass? couldn't get args to unpack
	hparams = HParams(**inspect.getargvalues(inspect.currentframe())[3])
	hparam_dict = hparams.values()
	print(hparam_dict)
	bot = crypto_bot.CryptoBot(hparams, test=TEST, tuning=True)
	cost = -bot.get_value()
	# TODO: figure out another method of logging because this can't be a part of the class
	"""if cost < self.lowest_cost:
		self.lowest_cost = cost
		crypto_bot.CryptoBot(hparams, tuning=True, dir_str='/best, hparam_str=hparam_str)"""
	return cost


"""
def gen_hparam_str(hparam_dict) -> str:
	hparam_str = '/hparams'
	for hparam, value in hparam_dict.items():
		hparam_str += (f'_{hparam}-{value}')
	return hparam_str

def gen_dir_str() -> str:
	timestamp = datetime.now().strftime('%Y%m%d_%H-%M-%S')
	dir_str = '/hparams_{0}'.format(timestamp)
	return dir_str
"""


class HyperparameterOptimizer:
	def __init__(self, test=False, hparam_dict=None, search_dim_dict=None):
		self.test = test
		self.test = test
		self.lowest_cost = 1
		if not (hparam_dict and search_dim_dict):
			self.hparam_dict = self.init_hparam_dict()
			self.search_space_dict = init_search_space_dict(test)

	def init_hparam_dict(self) -> dict:
		hparam_dict = set_hparams(self.test).values()
		return hparam_dict

	@property
	def hparams(self) -> HParams:
		hparams = HParams(**self.hparam_dict)
		return hparams

	@property
	def hparam_list(self) -> list:
		hparam_list = list(self.hparams.values().values())
		return hparam_list

	@property
	def search_space_list(self) -> list:
		search_space_list = list(self.search_space_dict.values())
		return search_space_list

	def optimize(self) -> dict:
		# TODO: optimize arguments to gp_minimize
		# Documentation: https://scikit-optimize.github.io/optimizer/index.html
		results = gp_minimize(func=run_bot,
							  dimensions=self.search_space_list,
							  acq_func='EI',  # could try something more exploratory or EIps for expected improvement per second
							  n_calls=100,
							  # we can change this to "EIps" for 'Expected improvement per second' to account for compute time
							  x0=self.hparam_list)  # we can mess with the default arguments later
		best_hparam_list = results.x
		hparam_space = results.space
		best_hparams = hparam_space.point_to_dict(best_hparam_list)
		self.best_hparams = best_hparams
		self.hparam_log = sorted(zip(results.func_vals, results.x_iters))
		return best_hparams


if __name__ == '__main__':
	main()
