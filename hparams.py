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
#import crypto_bot


def main():
	optimizer = HyperparameterOptimizer()
	results = optimizer.optimize()
	plot_convergence(results)
	print('Hyperparameeter optimization log:\n{optimizer.hparam_log}')
	print('Best hyperparameters:\n{optimizer.best_hparams}')


def set_hparams() -> HParams:
	hparams = HParams(
		# 20 coins (no including cash = BTC)
		 # high, open, volume, dp/dt
		window_size=50,
		stride=1,
		batch_size=100,
		num_training_steps=150000,
		learning_rate=2e-4,
		geometric_decay=0.0005,  # the large geometric_decay is the more recent times will be selected in training
		conv1_filter_length=3,
		num_conv1_features=2,
		num_conv2_features=20,
		num_fc1_neurons=12,
		dropout_keep_prob=0.5,
		conv_layers_separable=True,
		model_ending='two_fc_layers',
		batch_sampling_method='random_geometric',
		# start_time=None,
		# end_time=None #don't optimize
		# num_fc2_neurons = len(hparams.coin_pairs) + 1 (+1 needed for BTC)
	)
	return hparams


def set_params() -> HParams:
	params = HParams(coin_pairs = ["BTC_BTS", "BTC_ZEC", "BTC_STRAT", "BTC_XEM", "BTC_STEEM", "BTC_LTC", "BTC_ETC",
								   "BTC_XRP", "BTC_XMR", "BTC_DASH", "BTC_ETH", "BTC_STR", "BTC_LSK", "BTC_DOGE",
								   "BTC_SC", "BTC_SYS", "BTC_DGB", "BTC_MAID", "BTC_NXT", "BTC_BCN"],
					 num_input_channels=5,
					 live_bot_num_models=5,)
	params.add_hparam("num_coins", len(params.coin_pairs))
	return params


def init_search_space_dict() -> dict:
	search_space_dict = dict(
		dim_window_size=space.Integer(low=10, high=1000, name='window_size'),
		dim_stride=space.Integer(low=1, high=10, name='stride'),
		dim_batch_size=space.Integer(low=10, high=1000, name='batch_size'),
		dim_num_training_steps=space.Integer(low=10000, high=5000000, name='num_training_steps'),
		dim_learning_rate=space.Real(low=1e-6, high=1e-2, prior='log-uniform', name='learning_rate'),
		dim_geometric_decay=space.Real(low=1e-6, high=1, prior='log-uniform', name='geometric_decay'),
		conv1_filter_length=space.Integer(low=2, high=12, name='conv1_filter_length'),
		dim_num_conv1_features=space.Integer(low=1, high=64, name='num_conv1_features'),
		dim_num_conv2_features=space.Integer(low=8, high=128, name='num_conv2_features'),
		dim_num_fc1_neurons=space.Integer(low=8, high=32, name='num_fc1_neurons'),
		dim_dropout_keep_prob=space.Real(low=.1, high=.9, name='dropout_keep_prob'),
	)
	return search_space_dict


def init_search_space_list() -> list:
	search_space_list = list(init_search_space_dict().values())
	return search_space_list


@use_named_args(dimensions=init_search_space_list())
def run_bot(window_size,
			stride,
			batch_size,
			num_training_steps,
			learning_rate,
			geometric_decay,  # the large geometric_decay is the more recent times will be selected in training
			conv1_filter_length,
			num_conv1_features,
			num_conv2_features,
			num_fc1_neurons,
			dropout_keep_prob) -> float:
	hparams = HParams(**inspect.getargvalues(inspect.currentframe())[3])
	hparam_dict = hparams.values()
	hparam_str = gen_hparam_str(hparam_dict)
	print(hparam_str)
	bot = crypto_bot.CryptoBot(hparams, tuning=True, hparam_str=hparam_str)
	cost = -bot.get_value()
	# TODO: figure out another method of logging because this can't be a part of the class
	"""if cost < self.lowest_cost:
		self.lowest_cost = cost
		crypto_bot.CryptoBot(hparams, tuning=True, hparam_str='/best')"""
	return cost

def gen_hparam_str(hparam_dict) -> str:
	hparam_str = '/hparams'
	for hparam, value in hparam_dict.items():
		hparam_str += ('_{0}-{1}'.format(hparam, value))
	return hparam_str


class HyperparameterOptimizer:
	def __init__(self, hparam_dict=None, search_dim_dict=None):
		if not (hparam_dict and search_dim_dict):
			self.hparam_dict = self.init_hparam_dict()
			self.search_space_dict = init_search_space_dict()
		self.lowest_cost = 1

	@staticmethod
	def init_hparam_dict() -> dict:
		hparam_dict = dict(
			window_size=50,
			stride=1,
			batch_size=100,
			num_training_steps=150000,
			learning_rate=2e-4,
			geometric_decay=0.0005,  # the large geometric_decay is the more recent times will be selected in training
			conv1_filter_length=3,
			num_conv1_features=2,
			num_conv2_features=20,
			num_fc1_neurons=12,
			dropout_keep_prob=0.5,
		)
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
							  acq_func='EI',
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

