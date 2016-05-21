import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import time

def figsize(scalewidth, ratio = None):
	fig_width_pt = 426.79135                        # Get this from LaTeX using \the\textwidth
	in_per_pt = 1.0/72.27                           # Convert pt to inch
	if ratio is None:
	    ratio = (np.sqrt(5.0)-1.0)/2.0              # Aesthetic ratio (0.61803398875)
	fig_width = fig_width_pt*in_per_pt*scalewidth   # width in inches
	fig_height = fig_width*ratio                    # height in inches
	fig_size = [fig_width,fig_height]
	return fig_size

def plot():
	fig = plt.figure(figsize=figsize(1,0.5))

	x = np.arange(-2,8,0.001)
	vMorse = (1-np.exp(-x))**2
	vLin   = x**2

	plt.plot(x, vMorse, label = 'Morse potential')
	plt.plot(x, vLin, label = 'Taylor expanded Morse potential')

	plt.gca().set_xlabel('$ a (l-l_0) $')
	plt.gca().set_ylabel('$ v/D_e $')

	plt.gca().set_ylim(0,3)

	plt.legend(loc = 1, frameon = False)

	plt.tight_layout()

	if __name__ == '__main__':
		plt.show()

	return fig


if __name__ == '__main__':
	fig = plot()
	# fig.savefig('plot.png')

