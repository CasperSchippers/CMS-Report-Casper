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
	file = 'MDSLP5N2'
	data = pd.read_csv("Data\\" + file, header=None, delim_whitespace=True)

	noff = 4
	N = round((np.shape(data)[1]-noff)/3)

	plt.clf()
	fig = plt.figure(figsize=figsize(1,0.5))

	ax = plt.subplot(121, projection='3d')
	for n in range(N):
		ax.plot(data[3*n+noff], data[3*n+noff+1], data[3*n+noff+2])#'o', markeredgewidth=0.0, markersize=1)
	ax.view_init(20, 30)
	ax.set_xlabel('$x$', labelpad=-6)
	ax.set_ylabel('$y$', labelpad=-6)
	ax.set_zlabel('$z$', labelpad=2)
	for label in ax.get_zmajorticklabels():
		label.set_horizontalalignment('right')
	ax.tick_params(axis='x', pad=-4)
	ax.tick_params(axis='y', pad=-4)
	ax.tick_params(axis='z', pad=-3)
	ax.xaxis.set_ticks(np.arange(0, 1.1, 0.5))
	ax.yaxis.set_ticks(np.arange(-0.5, 0.6, 0.5))
	ax.zaxis.set_ticks(np.arange(-0.1, 0.11, 0.1))
	plt.tight_layout()

	plt.subplot(122)
	plt.plot(data[0],data[1], label='Kinetic energy')
	plt.plot(data[0],data[2], label='Potential energy')
	plt.plot(data[0],data[3], label='Total energy')
	plt.gca().set_xlabel('Time $t$')
	plt.gca().set_ylabel('Energy')
	plt.xlim(0,10)
	plt.ylim(-1.1,0.8)
	plt.legend(bbox_to_anchor=(1, 0.7), frameon=False)	
	plt.tight_layout()

	if __name__ == '__main__':
		plt.show()

	return fig


if __name__ == '__main__':
	fig = plot()
	# fig.savefig('plot.png')

