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
	fig = plt.figure(figsize=figsize(1,0.5)

	x = np.arange(0.001, 3, 0.01)
	y = 4*((x**-12) - (x**-6))

	plt.plot(x, y)
	plt.gca().set_xlabel('$r/\sigma$')
	plt.gca().set_ylabel('$ U_\mathrm{LJ}/\\varepsilon$')

	plt.gcf().subplots_adjust(bottom=0.15)

	plt.ylim(-1.5, 5)

	plt.tight_layout()

	if __name__ == '__main__':
		plt.show()

	return fig


if __name__ == '__main__':
	fig = plot()
	# fig.savefig('plot.png')

