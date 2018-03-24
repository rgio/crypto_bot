from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports of general code
import tensorflow as tf
import numpy as np
import pdb

def print_model_results(final_pvm, pvm, weights, path_to_model_dir, prefix):
	np.savetxt(path_to_model_dir + prefix + '_weights.out', weights, fmt='%.8f', delimiter=' ')
	np.savetxt(path_to_model_dir + prefix + '_value.out', pvm, fmt='%.8f', delimiter=' ')
	portfolio_value = np.ones(pvm.shape)
	portfolio_value[0] = pvm[0]
	for i in range(1,pvm.shape[0]):
		portfolio_value[i] = pvm[i] * portfolio_value[i-1]
	np.savetxt(path_to_model_dir + prefix + '_cumulative_value.out', portfolio_value, fmt='%.8f', delimiter=' ')
	print('The final %s portfolio multiplier is %g' % (prefix, final_pvm))


# TODO: FIX THIS
def print_hyperparameters(hparams, path_to_model_dir):
	# with open(path_to_model_dir+'hyperparameters.json', 'w') as f:
	# 	f.write(hparams.to_json())
	return