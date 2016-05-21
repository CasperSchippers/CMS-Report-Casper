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
	datafile = 'CHCl3-sol-298K-npt-6-rdf.xvg'
	data = pd.read_csv('Data/' + datafile, skiprows = 18, header = None, delim_whitespace = True)

	plt.clf()
	fig = plt.figure(figsize=figsize(1,0.5))

	ax = plt.subplot(121)
	ax.plot(data[0], data[1], label = 'Radial distribution function liquid')
	ax.set_xlabel('Distance [$\\mathrm{nm}$]')
	ax.set_ylabel('RDF')
	plt.legend(frameon = False)

	datafile = 'CHCl3-sol-298K-npt-heat-T508-rdf-7-8.xvg'
	data = pd.read_csv('Data/' + datafile, skiprows = 18, header = None, delim_whitespace = True)

	ax = plt.subplot(122)
	ax.plot(data[0], data[1], label = 'Radial distribution function gas')
	ax.set_xlabel('Distance [$\\mathrm{nm}$]')
	ax.set_ylabel('RDF')
	plt.legend(frameon = False)

	plt.tight_layout()

	if __name__ == '__main__':
		plt.show()

	return fig


if __name__ == '__main__':
	fig = plot()
	# fig.savefig('plot.png')

