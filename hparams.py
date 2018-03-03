from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports of general code
import tensorflow as tf

# Imports of our own code

def set_hyperparameters():
	hparams = tf.contrib.training.HParams(
		coin_pairs=["BTC_BTS","BTC_ZEC","BTC_STRAT","BTC_XEM","BTC_STEEM","BTC_LTC","BTC_ETC","BTC_XRP","BTC_XMR","BTC_DASH","BTC_ETH",
   				"BTC_STR", "BTC_LSK", "BTC_DOGE", "BTC_SC", "BTC_SYS", "BTC_DGB", "BTC_MAID", "BTC_NXT", "BTC_BCN"], # 20 coins (no including cash = BTC)
		window_size=50,
		stride=1,
		batch_size=100,
		batch_sampling_method='random_geometric', # options: random_geometric, random_uniform, systematic_uniform in price_data.py
		num_training_steps=200000,
		learning_rate=2e-4,
		geometric_decay=2.0, # larger geometric_decay leads to more recent times being selected more during training
		num_input_channels=4, # high, open, volume, dp/dt
		conv_layers_separable=True,
		len_conv1_filters=3,
		num_conv1_features=8,
		num_conv2_features=32,
		dropout_keep_prob=0.5,
		model_ending='one_fc_layer', # options: two_fc_layers, one_fc_layer, third_conv_layer in cnn.py
		num_fc1_neurons=12, # only for option two_fc_layers; it is set to num_coins for one_fc_layer in cnn.py
		len_conv3_filters=1, # only used in third_conv_layer option = 1 for 1x1 convolutions
		# num_fc2_neurons = len(hparams.coin_pairs) 
		)
	hparams.add_hparam("num_coins", len(hparams.coin_pairs))	
	return hparams

