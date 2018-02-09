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
		num_training_steps=200000,
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