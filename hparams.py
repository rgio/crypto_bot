from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports of general code
from tensorflow.contrib.training import HParams
from keras.callbacks import TensorBoard
from skopt import gp_minimize, forest_minimize
import skopt.space as space
from skopt.utils import use_named_args
from skopt.plots import plot_convergence

# local imports
import crypto_bot


def main():
	optimizer = HyperparameterOptimizer()
	results = optimizer.optimze()
	plot_convergence(results)


def set_parameters() -> HParams:
	hparams = HParams(
		coin_pairs=["BTC_BTS", "BTC_ZEC", "BTC_STRAT", "BTC_XEM", "BTC_STEEM", "BTC_LTC", "BTC_ETC", "BTC_XRP",
					"BTC_XMR", "BTC_DASH", "BTC_ETH",
					"BTC_STR", "BTC_LSK", "BTC_DOGE", "BTC_SC", "BTC_SYS", "BTC_DGB", "BTC_MAID", "BTC_NXT", "BTC_BCN"],
		# 20 coins (no including cash = BTC)
		num_input_channels=4,  # high, open, volume, dp/dt
		window_size=50,
		stride=1,
		batch_size=100,
		num_training_steps=150000,
		learning_rate=2e-4,
		geometric_decay=0.0005,  # the large geometric_decay is the more recent times will be selected in training
		conv1_filter_length=3,
		# conv1_filter_length=3,
		num_conv1_features=2,
		num_conv2_features=20,
		num_fc1_neurons=12,
		dropout_keep_prob=0.5,
		# start_time=None,
		# end_time=None #don't optimize
		# num_fc2_neurons = len(hparams.coin_pairs) + 1 (+1 needed for BTC)
	)
	hparams.add_hparam("num_coins", len(hparams.coin_pairs))
	return hparams


def initialize_search_space_dimension_dict() -> dict:
	search_dim_dict = dict(
		dim_window_size=space.Integer(low=10, high=1000, name='window_size'),
		dim_stride=space.Integer(low=1, high=10, name='stride'),
		dim_batch_size=space.Integer(low=10, high=1000, name='batch_size'),
		dim_num_training_steps=space.Integer(low=10000, high=5000000, name='num_training_steps'),
		dim_learning_rate=space.Real(low=1e-6, high=1e-2, prior='log-uniform', name='learning_rate'),
		dim_geometric_decay=space.Real(low=0, high=1, prior='log-uniform', name='geometric_decay'),
		conv1_filter_length=space.Integer(low=2, high=12, name='conv1_filter_length'),
		# conv2_filter_length=space.Integer(low=8, high=self.hparams['window_size']-2, name='conv2_filter_length'),
		dim_num_conv1_features=space.Integer(low=1, high=64, name='num_conv1_features'),
		dim_num_conv2_features=space.Integer(low=8, high=128, name='num_conv2_features'),
		dim_num_fc1_neurons=space.Integer(low=8, high=32, name='num_fc1_neurons'),
		dim_dropout_keep_prob=space.Real(low=.1, high=.9, name='dropout_keep_prob'),
		# dim_start_time=space.Integer(low=8, high=32, name='start_time'),
	)
	return search_dim_dict


class HyperparameterOptimizer:
	def __init__(self, hparam_dict=None, search_dim_dict=None):
		if not (hparam_dict and search_dim_dict):
			self.hparam_dict = self.init_hparam_dict()
			self.hparam_list = list(self.hparam_dict.values())
			self.search_dim_dict = initialize_search_space_dimension_dict()
			self.search_dim_list = list(self.search_dim_dict.values())
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
			# conv1_filter_length=3,
			num_input_channels=4,  # high, open, volume, dp/dt
			num_conv1_features=2,
			num_conv2_features=20,
			num_fc1_neurons=12,
			dropout_keep_prob=0.5,
			# start_time=None,
			# end_time=None #don't optimize
			# num_fc2_neurons = len(hparams.coin_pairs) + 1 (+1 needed for BTC)
		)
		return hparam_dict

	@property
	def hparams(self) -> HParams:
		hparams = HParams(**self.hparam_dict)
		return hparams

	@property
	def hparam_defaults(self):
		hparams = HParams(**self.init_hparam_dict())
		return hparams

	@property
	def hparam_str(self) -> str:
		hparam_str = '/hparams'
		for hparam, value in self.hparam_dict:
			hparam_str.append(f'_{hparam}-{value}')
		return hparam_str

	@use_named_args(dimensions=list(initialize_search_space_dimension_dict().values()))
	def run_bot(self, **hyperparameters) -> float:
		# TODO: add this as an optional argument to crypto_bot
		hparams = HParams(**hyperparameters)
		bot = crypto_bot.CryptoBot(hparams, tuning=True, hparam_str=self.hparam_str)
		cost = 1 - bot.get_cost()
		if cost < self.lowest_cost:
			self.lowest_cost = cost
			crypto_bot.CryptoBot(hparams, tuning=True, hparam_str='/best')
		return cost

	def optimize(self) -> dict:
		# We can optimize more of the arguments to gp_minimize later.
		# Documentation: https://scikit-optimize.github.io/optimizer/index.html
		results = gp_minimize(func=self.run_bot,
							  dimensions=self.search_dim_list,
							  acq_func='EI',
							  # we can change this to "EIps" for 'Expected improvement per second' to account for compute time
							  x0=self.hparam_defaults)  # we can mess with the default arguments later
		best_hparam_list = results.x
		hparam_space = results.space
		best_hparams = hparam_space.point_to_dict(best_hparam_list)
		self.best_hparams = best_hparams
		return best_hparams


if __name__ == '__main__':
	main()
