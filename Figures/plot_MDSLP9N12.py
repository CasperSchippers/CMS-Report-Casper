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
	file = 'MDSLP9N12'
	data = pd.read_csv("Data\\" + file, header=None, delim_whitespace=True)

	noff = 4
	N = round((np.shape(data)[1]-noff)/6)

	plt.clf()
	fig = plt.figure(figsize=figsize(1,0.5))

	ax = plt.subplot(121, projection='3d')
	ax.set_rasterization_zorder(1);
	for n in range(N):
		ax.plot(data[6*n+noff], data[6*n+noff+1], data[6*n+noff+2],'o', markeredgewidth=0.0, markersize=1,zorder=0)
	ax.view_init(20, 30)
	ax.set_xlabel('$x$', labelpad=-6)
	ax.set_ylabel('$y$', labelpad=-6)
	ax.set_zlabel('$z$', labelpad=-4)
	for label in ax.get_zmajorticklabels():
		label.set_horizontalalignment('right')
	ax.tick_params(axis='x', pad=-4)
	ax.tick_params(axis='y', pad=-4)
	ax.tick_params(axis='z', pad=-3)
	ax.xaxis.set_ticks(np.arange(-4, 5, 2))
	ax.yaxis.set_ticks(np.arange(-4, 5, 2))
	ax.zaxis.set_ticks(np.arange(-8, 9, 2))
	plt.tight_layout()

	ax = plt.subplot(122, projection='3d')
	ax.set_rasterization_zorder(1);
	for n in range(N):
		ax.plot(data[6*n+noff+3], data[6*n+noff+4], data[6*n+noff+5],'o', markeredgewidth=0.0, markersize=1,zorder=0)
	ax.view_init(20, 30)
	ax.set_xlabel('$x$', labelpad=-6)
	ax.set_ylabel('$y$', labelpad=-6)
	ax.set_zlabel('$z$', labelpad=-4)
	for label in ax.get_zmajorticklabels():
		label.set_horizontalalignment('right')
	ax.tick_params(axis='x', pad=-4)
	ax.tick_params(axis='y', pad=-4)
	ax.tick_params(axis='z', pad=-3)
	ax.xaxis.set_ticks(np.arange(0,4.1,1))
	ax.yaxis.set_ticks(np.arange(0,4.1,1))
	ax.zaxis.set_ticks(np.arange(0,4.1,1))
	plt.tight_layout()

	if __name__ == '__main__':
		plt.show()

	return fig


if __name__ == '__main__':
	fig = plot()
	# fig.savefig('plot.png')

