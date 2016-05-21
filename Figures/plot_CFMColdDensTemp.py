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
	datafile = 'CHCl3-sol-298K-npt-6-temp-dens.xvg'

	plt.clf()
	fig = plt.figure(figsize=figsize(0.9,0.5))
	
	
	ax1 = plt.gca()	
	ax2 = ax1.twinx()
	ax1.set_zorder(ax2.get_zorder()+1) # put ax in front of ax2 
	ax1.patch.set_visible(False) # hide the 'canvas' 
	data = pd.read_csv('Data/' + datafile, skiprows = 23, header = None, delim_whitespace = True)

	ln1 = ax1.plot(data[0]/1000,data[2],'b-', label = 'Density')
	ax1.set_xlabel('Time [$\\mathrm{ns}$]')
	ax1.set_ylabel('Density [$\\mathrm{kg/m^3}$]')
	ax1.set_ylim(0, 1600)

	ln2 = ax2.plot(data[0]/1000,data[1],'r-', linewidth = 0.00001, label = 'Temperature')
	ax2.set_xlabel('Time [$\\mathrm{ns}$]')
	ax2.set_ylabel('Temperature [$\\mathrm{K}$]')
	ax2.set_ylim(250, 550)
	ax1.set_xlim(0,0.10)

	lns = ln1 + ln2
	lbs = [l.get_label() for l in lns]
	plt.legend(lns, lbs, loc = 7, frameon = False)

	plt.tight_layout()

	if __name__ == '__main__':
		plt.show()

	return fig


if __name__ == '__main__':
	fig = plot()
	# fig.savefig('plot.png')

