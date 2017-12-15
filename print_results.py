from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports of general code
import numpy as np
import pdb

def print_model_results(final_pvm, pvm, weights, path_to_model_dir):
	np.savetxt(path_to_model_dir + 'weights.out', weights, fmt='%.8f', delimiter=' ')
	np.savetxt(path_to_model_dir + 'value.out', pvm, fmt='%.8f', delimiter=' ')
	print('The final portfolio multiplier is %g' % final_pvm)
