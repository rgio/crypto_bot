from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports of general code
import tensorflow as tf
import skopt.space as space

# Imports of our own code

def set_hyperparameters():
	hparams = tf.contrib.training.HParams(
		coin_pairs=["BTC_BTS","BTC_ZEC","BTC_STRAT","BTC_XEM","BTC_STEEM","BTC_LTC","BTC_ETC","BTC_XRP","BTC_XMR","BTC_DASH","BTC_ETH",
   				"BTC_STR", "BTC_LSK", "BTC_DOGE", "BTC_SC", "BTC_SYS", "BTC_DGB", "BTC_MAID", "BTC_NXT", "BTC_BCN"], # 20 coins (no including cash = BTC)
		window_size=50,
		stride=1,
		batch_size=100,
		num_training_steps=150000,
		learning_rate=2e-4,
		geometric_decay=2.0, # the large geometric_decay is the more recent times will be selected in training
		num_input_channels=4, # high, open, volume, dp/dt
		num_conv1_features=2,
		num_conv2_features=20,
		num_fc1_neurons=12,
		dropout_keep_prob=0.5,
		# num_fc2_neurons = len(hparams.coin_pairs) + 1 (+1 needed for BTC)
		)
	hparams.add_hparam("num_coins", len(hparams.coin_pairs))
	return hparams

class HyperparameterOptimzer:
	def __init__(self, hparam_dict=None, search_dims=None):
		if not (hparam_dict or search_dims):
			self.hparam_dict = self.initialize_hyperparameters()
			self.search_dims = self.initialize_search_space_dimensions()

	@staticmethod
	def initialize_hyperparameters() -> dict:
		hparam_dict = dict(
			window_size=50,
			stride=1,
			batch_size=100,
			num_training_steps=150000,
			learning_rate=2e-4,
			geometric_decay=2.0,  # the large geometric_decay is the more recent times will be selected in training
			num_input_channels=4,  # high, open, volume, dp/dt
			num_conv1_features=2,
			num_conv2_features=20,
			num_fc1_neurons=12,
			dropout_keep_prob=0.5,
			# num_fc2_neurons = len(hparams.coin_pairs) + 1 (+1 needed for BTC)
		)
		return hparam_dict

	@staticmethod
	def initialize_search_space_dimensions() -> dict:
		search_dims = dict(
			dim_window_size=space.Integer(low=10, high=1000, name='window_size'),
			dim_stride=space.Integer(low=1, high=10, name='stride'),
			dim_batch_size=space.Integer(low=10, high=1000, name='batch_size'),
			dim_num_training_steps=space.Integer(low=10000, high=5000000, name='num_training_steps'),
			dim_learning_rate=space.Real(low=1e-6, high=1e-2, prior='log-uniform', name='learning_rate'),
			dim_geometric_decay=space.Real('''IDK''', prior='log-uniform', name='geometric_decay'), # TODO: I thought this was from 0 -> -1, so correct this if I'm wrong
			dim_num_input_channels=space.Integer(), # TODO need to fill these in
			dim_num_conv1_features=space.Integer(),
			dim_num_conv2_features=space.Integer(),
			dim_num_fc1_neurons=space.Integer(),
			dim_dropout_keep_prob=space.Real(low=.1, high=.9, name='dropout_keep_prob'),
			# num_fc2_neurons = len(hparams.coin_pairs) + 1 (+1 needed for BTC)
		)
		return search_dims
