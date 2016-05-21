import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import time
from multiprocessing import Pool, Process, Lock, Queue

# defining standard plotting functions
def figsize(scalewidth, ratio = None):
    fig_width_pt = 426.79135                        # Get this from LaTeX using \the\textwidth
    in_per_pt = 1.0/72.27                           # Convert pt to inch
    if ratio is None:
        ratio = (np.sqrt(5.0)-1.0)/2.0              # Aesthetic ratio (0.61803398875)
    fig_width = fig_width_pt*in_per_pt*scalewidth   # width in inches
    fig_height = fig_width*ratio                    # height in inches
    fig_size = [fig_width,fig_height]
    return fig_size

def newfig(widthscale,ratio = None):
    plt.clf()
    fig = plt.figure(figsize=figsize(widthscale, ratio))
    ax = fig.add_subplot(111)
    return fig, ax

def savefig(filename):
    plt.savefig('%s.pgf' % filename, dpi = 2000)
    plt.savefig('%s.pdf' % filename, dpi = 2000)
    # plt.savefig('%s.png' % filename)

def worker(q,l):
	f = q.get()
	while not f == "STOP":
		f()
		f = q.get()

# mpl settings
mpl.use('pgf')

pgf_with_latex = {
    "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
    "text.usetex": True,                # use LaTeX to write all text
    "font.family": "serif",
    "font.serif": [],                   # blank entries should cause plots to inherit fonts from the document
    "font.sans-serif": [],
    "font.monospace": [],
    "axes.labelsize": 10,               # LaTeX default is 10pt font.
    "font.size": 10,
    "legend.fontsize": 8,               # Make the legend/label fonts a little smaller
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.figsize": figsize(0.9),     # default fig size of 0.9 textwidth
    "pgf.preamble": [
        r"\usepackage[utf8x]{inputenc}",    # use utf8 fonts becasue your computer can handle it :)
        r"\usepackage[T1]{fontenc}",        # plots will be generated using this preamble
        ],
    "pgf.rcfonts": False                # use LaTeX to render the text with the rest of the LaTeX file
}

mpl.rcParams.update(pgf_with_latex)

# import pyplot after setting mpl
import matplotlib.pyplot as plt

def MDSLEX2b():
	start = time.clock()
	file = 'MDSLEX2b'

	plt.figure(figsize=figsize(0.7, 0.6))

	x = np.arange(0.001, 3, 0.01)
	y = 4*((x**-12) - (x**-6))

	plt.plot(x, y)
	plt.gca().set_xlabel('$r/\sigma$')
	plt.gca().set_ylabel('$ U_\mathrm{LJ}/\\varepsilon$')

	plt.gcf().subplots_adjust(bottom=0.15)

	plt.ylim(-1.5, 5)


	savefig(file)
	end = time.clock()
	print('Created figure \'%s\' in %f s' % (file, end-start))

def MDSLP1():
	start = time.clock()
	file = "MDSLP1"
	data = pd.read_csv("Data\\" + file, header=None, delim_whitespace=True)

	plt.clf()
	plt.figure(figsize=figsize(1,0.5))

	ax = plt.subplot(121, projection='3d')
	ax.plot(data[4], data[5], data[6])
	ax.plot(data[7], data[8], data[9])
	ax.view_init(20, 30)
	ax.set_xlabel('$x$', labelpad=-6)
	ax.set_ylabel('$y$', labelpad=-6)
	ax.set_zlabel('$z$', labelpad=-4)
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
	plt.ylim(-1.05,0.8)
	plt.xlim(0,10)
	plt.legend(bbox_to_anchor=(1, 0.8), frameon=False)
	plt.tight_layout()

	savefig(file)
	end = time.clock()
	print('Created figure \'%s\' in %f s' % (file, end-start))

def MDSLP2():
	start = time.clock()
	file = "MDSLP2"
	data = pd.read_csv("Data\\" + file, header=None, delim_whitespace=True)

	plt.clf()
	plt.figure(figsize=figsize(1,0.5))

	ax = plt.subplot(121, projection='3d')
	ax.plot(data[4], data[5], data[6])
	ax.plot(data[7], data[8], data[9])
	ax.plot(data[10], data[11], data[12])
	ax.view_init(20, 30)
	ax.set_xlabel('$x$', labelpad=-6)
	ax.set_ylabel('$y$', labelpad=-6)
	ax.set_zlabel('$z$', labelpad=-4)
	for label in ax.get_zmajorticklabels():
		label.set_horizontalalignment('right')
	ax.tick_params(axis='x', pad=-4)
	ax.tick_params(axis='y', pad=-4)
	ax.tick_params(axis='z', pad=-3)
	ax.xaxis.set_ticks(np.arange(-0.5, 1.6, 0.5))
	ax.yaxis.set_ticks(np.arange(-0.5, 1.6, 0.5))
	ax.zaxis.set_ticks(np.arange(-0.6, 0.9, 0.2))
	plt.tight_layout()

	plt.subplot(122)
	plt.plot(data[0],data[1], label='Kinetic energy')
	plt.plot(data[0],data[2], label='Potential energy')
	plt.plot(data[0],data[3], label='Total energy')
	plt.gca().set_xlabel('Time $t$')
	plt.gca().set_ylabel('Energy')
	plt.ylim(-3,2)
	plt.xlim(0,10)
	plt.legend(bbox_to_anchor=(1, 0.7), frameon=False)	
	plt.tight_layout()

	savefig(file)
	end = time.clock()
	print('Created figure \'%s\' in %f s' % (file, end-start))

def MDSLP3N4():
	start = time.clock()
	file = "MDSLP3N4"
	data = pd.read_csv("Data\\" + file, header=None, delim_whitespace=True)

	plt.clf()
	plt.figure(figsize=figsize(1,0.5))

	ax = plt.subplot(121, projection='3d')
	ax.plot(data[4], data[5], data[6])
	ax.plot(data[7], data[8], data[9])
	ax.plot(data[10], data[11], data[12])
	ax.plot(data[13], data[14], data[15])
	ax.view_init(20, 30)
	ax.set_xlabel('$x$', labelpad=-6)
	ax.set_ylabel('$y$', labelpad=-6)
	ax.set_zlabel('$z$', labelpad=-4)
	for label in ax.get_zmajorticklabels():
		label.set_horizontalalignment('right')
	ax.tick_params(axis='x', pad=-4)
	ax.tick_params(axis='y', pad=-4)
	ax.tick_params(axis='z', pad=-3)
	ax.xaxis.set_ticks(np.arange(-2, 5.1, 2))
	ax.yaxis.set_ticks(np.arange(-1, 3.1, 1))
	ax.zaxis.set_ticks(np.arange(-2, 2.1, 1))
	plt.tight_layout()

	plt.subplot(122)
	plt.plot(data[0],data[1], label='Kinetic energy')
	plt.plot(data[0],data[2], label='Potential energy')
	plt.plot(data[0],data[3], label='Total energy')
	plt.gca().set_xlabel('Time $t$')
	plt.gca().set_ylabel('Energy')
	plt.xlim(0,10)
	plt.legend(bbox_to_anchor=(1, 0.7), frameon=False)	
	plt.tight_layout()

	savefig(file)
	end = time.clock()
	print('Created figure \'%s\' in %f s' % (file, end-start))

def MDSLP3N5():
	start = time.clock()
	file = "MDSLP3N5"
	data = pd.read_csv("Data\\" + file, header=None, delim_whitespace=True)

	plt.clf()
	plt.figure(figsize=figsize(1,0.5))

	ax = plt.subplot(121, projection='3d')
	ax.plot(data[4], data[5], data[6])
	ax.plot(data[7], data[8], data[9])
	ax.plot(data[10], data[11], data[12])
	ax.plot(data[13], data[14], data[15])
	ax.plot(data[16], data[17], data[18])
	ax.view_init(20, 30)
	ax.set_xlabel('$x$', labelpad=-6)
	ax.set_ylabel('$y$', labelpad=-6)
	ax.set_zlabel('$z$', labelpad=-4)
	for label in ax.get_zmajorticklabels():
		label.set_horizontalalignment('right')
	ax.tick_params(axis='x', pad=-4)
	ax.tick_params(axis='y', pad=-4)
	ax.tick_params(axis='z', pad=-3)
	ax.xaxis.set_ticks(np.arange(-1, 2.1, 1))
	ax.yaxis.set_ticks(np.arange(-1, 2.1, 1))
	ax.zaxis.set_ticks(np.arange(-1, 2.5, 1))
	plt.tight_layout()

	plt.subplot(122)
	plt.plot(data[0],data[1], label='Kinetic energy')
	plt.plot(data[0],data[2], label='Potential energy')
	plt.plot(data[0],data[3], label='Total energy')
	plt.gca().set_xlabel('Time $t$')
	plt.gca().set_ylabel('Energy')
	plt.xlim(0,10)
	plt.legend(bbox_to_anchor=(1, 0.7), frameon=False)	
	plt.tight_layout()

	savefig(file)
	end = time.clock()
	print('Created figure \'%s\' in %f s' % (file, end-start))

def MDSLP3N6():
	start = time.clock()
	file = "MDSLP3N6"
	data = pd.read_csv("Data\\" + file, header=None, delim_whitespace=True)

	plt.clf()
	plt.figure(figsize=figsize(1,0.5))

	ax = plt.subplot(121, projection='3d')
	ax.plot(data[4], data[5], data[6])
	ax.plot(data[7], data[8], data[9])
	ax.plot(data[10], data[11], data[12])
	ax.plot(data[13], data[14], data[15])
	ax.plot(data[16], data[17], data[18])
	ax.view_init(20, 30)
	ax.set_xlabel('$x$', labelpad=-6)
	ax.set_ylabel('$y$', labelpad=-6)
	ax.set_zlabel('$z$', labelpad=-4)
	for label in ax.get_zmajorticklabels():
		label.set_horizontalalignment('right')
	ax.tick_params(axis='x', pad=-4)
	ax.tick_params(axis='y', pad=-4)
	ax.tick_params(axis='z', pad=-3)
	ax.xaxis.set_ticks(np.arange(-0, 2.5, 1))
	ax.yaxis.set_ticks(np.arange(-1, 2.1, 1))
	ax.zaxis.set_ticks(np.arange(0, 2.5, 1))
	plt.tight_layout()

	plt.subplot(122)
	plt.plot(data[0],data[1], label='Kinetic energy')
	plt.plot(data[0],data[2], label='Potential energy')
	plt.plot(data[0],data[3], label='Total energy')
	plt.gca().set_xlabel('Time $t$')
	plt.gca().set_ylabel('Energy')
	plt.xlim(0,10)
	plt.legend(bbox_to_anchor=(1, 0.7), frameon=False)	
	plt.tight_layout()

	savefig(file)
	end = time.clock()
	print('Created figure \'%s\' in %f s' % (file, end-start))

def MDSLP5N2():
	start = time.clock()
	file = 'MDSLP5N2'
	data = pd.read_csv("Data\\" + file, header=None, delim_whitespace=True)

	noff = 4
	N = round((np.shape(data)[1]-noff)/6)

	plt.clf()
	plt.figure(figsize=figsize(1,0.5))

	ax = plt.subplot(121, projection='3d')
	for n in range(N):
		ax.plot(data[6*n+noff], data[6*n+noff+1], data[6*n+noff+2])#'o', markeredgewidth=0.0, markersize=1)
	ax.view_init(20, 30)
	ax.set_xlabel('$x$', labelpad=-6)
	ax.set_ylabel('$y$', labelpad=-6)
	ax.set_zlabel('$z$', labelpad=-4)
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

	savefig(file)
	end = time.clock()
	print('Created figure \'%s\' in %f s' % (file, end-start))

def MDSLP5N3():
	start = time.clock()
	file = 'MDSLP5N3'
	data = pd.read_csv("Data\\" + file, header=None, delim_whitespace=True)

	noff = 4
	N = round((np.shape(data)[1]-noff)/6)

	plt.clf()
	plt.figure(figsize=figsize(1,0.5))

	ax = plt.subplot(121, projection='3d')
	for n in range(N):
		ax.plot(data[6*n+noff], data[6*n+noff+1], data[6*n+noff+2])#'o', markeredgewidth=0.0, markersize=1)
	ax.view_init(20, 30)
	ax.set_xlabel('$x$', labelpad=-6)
	ax.set_ylabel('$y$', labelpad=-6)
	ax.set_zlabel('$z$', labelpad=-4)
	for label in ax.get_zmajorticklabels():
		label.set_horizontalalignment('right')
	ax.tick_params(axis='x', pad=-4)
	ax.tick_params(axis='y', pad=-4)
	ax.tick_params(axis='z', pad=-3)
	ax.xaxis.set_ticks(np.arange(0, 1.1, 0.5))
	ax.yaxis.set_ticks(np.arange(-1, 1.1, 0.5))
	ax.zaxis.set_ticks(np.arange(-0.6, 0.9, 0.2))
	plt.tight_layout()

	plt.subplot(122)
	plt.plot(data[0],data[1], label='Kinetic energy')
	plt.plot(data[0],data[2], label='Potential energy')
	plt.plot(data[0],data[3], label='Total energy')
	plt.gca().set_xlabel('Time $t$')
	plt.gca().set_ylabel('Energy')
	plt.xlim(0,10)
	plt.ylim(-3,2)
	plt.legend(bbox_to_anchor=(1, 0.7), frameon=False)	
	plt.tight_layout()

	savefig(file)
	end = time.clock()
	print('Created figure \'%s\' in %f s' % (file, end-start))

def MDSLP8N12():
	start = time.clock()
	file = 'MDSLP8N12'
	data = pd.read_csv("Data\\" + file, header=None, delim_whitespace=True)

	noff = 4
	N = round((np.shape(data)[1]-noff)/6)

	plt.clf()
	plt.figure(figsize=figsize(1,0.5))

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

	plt.subplot(122)
	plt.plot(data[0],data[1], label='Kinetic energy')
	plt.plot(data[0],data[2], label='Potential energy')
	plt.plot(data[0],data[3], label='Total energy')
	plt.gca().set_xlabel('Time $t$')
	plt.gca().set_ylabel('Energy')
	plt.xlim(0,10)
	plt.legend(bbox_to_anchor=(1, 0.7), frameon=False)	
	plt.tight_layout()

	savefig(file)
	end = time.clock()
	print('Created figure \'%s\' in %f s' % (file, end-start))

def MDSLP8N24():
	start = time.clock()
	file = 'MDSLP8N24'
	data = pd.read_csv("Data\\" + file, header=None, delim_whitespace=True)

	noff = 4
	N = round((np.shape(data)[1]-noff)/6)

	plt.clf()
	plt.figure(figsize=figsize(1,0.5))

	ax = plt.subplot(121, projection='3d')
	ax.set_rasterization_zorder(1);
	for n in range(N):
		ax.plot(data[6*n+noff], data[6*n+noff+1], data[6*n+noff+2],'o', markeredgewidth=0.0, markersize=1, zorder = 0)
	ax.view_init(20, 30)
	ax.set_xlabel('$x$', labelpad=-6)
	ax.set_ylabel('$y$', labelpad=-6)
	ax.set_zlabel('$z$', labelpad=-4)
	for label in ax.get_zmajorticklabels():
		label.set_horizontalalignment('right')
	ax.tick_params(axis='x', pad=-4)
	ax.tick_params(axis='y', pad=-4)
	ax.tick_params(axis='z', pad=-3)
	ax.xaxis.set_ticks(np.arange(-20, 31, 10))
	ax.yaxis.set_ticks(np.arange(-20, 21, 10))
	ax.zaxis.set_ticks(np.arange(-20, 31, 10))
	# ax.set_rasterization_zorder(-10)
	plt.tight_layout()

	plt.subplot(122)
	plt.plot(data[0],data[1], label='Kinetic energy')
	plt.plot(data[0],data[2], label='Potential energy')
	plt.plot(data[0],data[3], label='Total energy')
	plt.gca().set_xlabel('Time $t$')
	plt.gca().set_ylabel('Energy')
	plt.xlim(0,10)
	plt.legend(bbox_to_anchor=(1, 0.7), frameon=False)	
	plt.tight_layout()

	savefig(file)
	end = time.clock()
	print('Created figure \'%s\' in %f s' % (file, end-start))

def ArgonAnnealedDensTemp():
	start = time.clock()
	file = 'ArgonAnnealedDensTemp'
	datafile = 'argon_start_1000_85K_npt-heat-temp-dens.xvg'

	plt.clf()
	plt.figure(figsize=figsize(0.9,0.5))
	
	
	ax1 = plt.gca()	
	ax2 = ax1.twinx()
	ax1.set_zorder(ax2.get_zorder()+1) # put ax in front of ax2 
	ax1.patch.set_visible(False) # hide the 'canvas' 
	data = pd.read_csv('Data/' + datafile, skiprows = 23, header = None, delim_whitespace = True)

	ln1 = ax1.plot(data[0]/1000,data[2],'b-', label = 'Density')
	ax1.set_xlabel('Time [$\\mathrm{ns}$]')
	ax1.set_ylabel('Density [$\\mathrm{kg/m^3}$]')

	ln2 = ax2.plot(data[0]/1000,data[1],'r-', linewidth = 0.00001, label = 'Temperature')
	ax2.set_xlabel('Time [$\\mathrm{ns}$]')
	ax2.set_ylabel('Temperature [$\\mathrm{K}$]')

	lns = ln1 + ln2
	lbs = [l.get_label() for l in lns]
	plt.legend(lns, lbs, loc = 7, frameon = False)

	plt.tight_layout()

	savefig(file)
	end = time.clock()
	print('Created figure \'%s\' in %f s' % (file, end-start))

def ArgonColdDensTemp():
	start = time.clock()
	file = 'ArgonColdDensTemp'
	datafile = 'argon_start_1000_85K_npt-temp-density.xvg'

	plt.clf()
	plt.figure(figsize=figsize(0.9,0.5))
	
	
	ax1 = plt.gca()	
	ax2 = ax1.twinx()
	ax1.set_zorder(ax2.get_zorder()+1) # put ax in front of ax2 
	ax1.patch.set_visible(False) # hide the 'canvas' 
	data = pd.read_csv('Data/' + datafile, skiprows = 23, header = None, delim_whitespace = True)

	ln1 = ax1.plot(data[0]/1000,data[2],'b-', label = 'Density')
	ax1.set_xlabel('Time [$\\mathrm{ns}$]')
	ax1.set_ylabel('Density [$\\mathrm{kg/m^3}$]')
	ax1.set_ylim(0, 1400)

	ln2 = ax2.plot(data[0]/1000,data[1],'r-', linewidth = 0.00001, label = 'Temperature')
	ax2.set_xlabel('Time [$\\mathrm{ns}$]')
	ax2.set_ylabel('Temperature [$\\mathrm{K}$]')
	ax2.set_ylim(80, 125)
	# ax1.set_xlim(0,0.10)

	lns = ln1 + ln2
	lbs = [l.get_label() for l in lns]
	plt.legend(lns, lbs, loc = 7, frameon = False)

	plt.tight_layout()

	savefig(file)
	end = time.clock()
	print('Created figure \'%s\' in %f s' % (file, end-start))


def ArgonMSD():
	start = time.clock()
	file = 'ArgonMSD'
	datafile = 'argon_start_1000_85K_npt-msd.xvg'
	data = pd.read_csv('Data/' + datafile, skiprows = 18, header = None, delim_whitespace = True)

	plt.clf()
	plt.figure(figsize=figsize(1,0.5))

	ax = plt.subplot(121)
	ax.plot(data[0]/1000, data[1], label = 'Simulation liquid')
	ax.set_xlabel('Time [$\\mathrm{ns}$]')
	ax.set_ylabel('MSD [$\\mathrm{nm^2}$]')
	ax.autoscale(False)
	
	fit = np.polyfit(data[0], data[1], 1)
	p2 = np.poly1d(fit)
	ax.plot(data[0]/1000, p2(data[0]),'r-', label = 'Linear fit')
	plt.legend(loc = 2,frameon = False)

	datafile = 'argon_start_1000_85K_npt-heat-msd.xvg'
	data = pd.read_csv('Data/' + datafile, skiprows = 18, header = None, delim_whitespace = True)

	ax = plt.subplot(122)
	ax.plot(data[0]/1000+6, data[1], label='Simulation gas')
	ax.set_xlabel('Time [$\\mathrm{ns}$]')
	ax.set_ylabel('MSD [$\\mathrm{nm^2}$]')
	ax.autoscale(False)
	
	fitdata = data[:]
	fit = np.polyfit(fitdata[0], fitdata[1], 1)
	p1 = np.poly1d(fit)
	ax.plot(data[0]/1000+6, p1(data[0]),'r-', label = 'Linear fit')
	plt.legend(loc = 2,frameon = False)

	plt.tight_layout()

	savefig(file)
	end = time.clock()
	print('Created figure \'%s\' in %f s' % (file, end-start), ', diffusion coefficient liquid = %.4e' % (p2[1]/6), ', diffusion coefficient gas = %.4e' % (p1[1]/6))

def ArgonRDF():
	start = time.clock()
	file = 'ArgonRDF'
	datafile = 'argon_start_1000_85K_npt-rdf.xvg'
	data = pd.read_csv('Data/' + datafile, skiprows = 18, header = None, delim_whitespace = True)

	plt.clf()
	plt.figure(figsize=figsize(1,0.5))

	ax = plt.subplot(121)
	ax.plot(data[0], data[1], label = 'Radial distribution function liquid')
	ax.set_xlabel('Distance [$\\mathrm{nm}$]')
	ax.set_ylabel('RDF')
	plt.legend(frameon = False)

	datafile = 'argon_start_1000_85K_npt-heat-rdf.xvg'
	data = pd.read_csv('Data/' + datafile, skiprows = 18, header = None, delim_whitespace = True)

	ax = plt.subplot(122)
	ax.plot(data[0], data[1], label = 'Radial distribution function gas')
	ax.set_xlabel('Distance [$\\mathrm{nm}$]')
	ax.set_ylabel('RDF')
	plt.legend(frameon = False)

	plt.tight_layout()

	savefig(file)
	end = time.clock()
	print('Created figure \'%s\' in %f s' % (file, end-start))

def CFMAnnealedDensTemp():
	start = time.clock()
	file = 'CFMAnnealedDensTemp'
	datafile = 'CHCl3-sol-298K-npt-heat-T508-temp-dens.xvg'

	plt.clf()
	plt.figure(figsize=figsize(0.9,0.5))
	
	
	ax1 = plt.gca()	
	ax2 = ax1.twinx()
	ax1.set_zorder(ax2.get_zorder()+1) # put ax in front of ax2 
	ax1.patch.set_visible(False) # hide the 'canvas' 
	data = pd.read_csv('Data/' + datafile, skiprows = 23, header = None, delim_whitespace = True)

	ln1 = ax1.plot(data[0]/1000,data[2],'b-', label = 'Density')
	ax1.set_xlabel('Time [$\\mathrm{ns}$]')
	ax1.set_ylabel('Density [$\\mathrm{kg/m^3}$]')

	ln2 = ax2.plot(data[0]/1000,data[1],'r-', linewidth = 0.00001, label = 'Temperature')
	ax2.set_xlabel('Time [$\\mathrm{ns}$]')
	ax2.set_ylabel('Temperature [$\\mathrm{K}$]')

	lns = ln1 + ln2
	lbs = [l.get_label() for l in lns]
	plt.legend(lns, lbs, loc = 7, frameon = False)

	plt.tight_layout()

	savefig(file)
	end = time.clock()
	print('Created figure \'%s\' in %f s' % (file, end-start))

def CFMColdDensTemp():
	start = time.clock()
	file = 'CFMColdDensTemp'
	datafile = 'CHCl3-sol-298K-npt-6-temp-dens.xvg'

	plt.clf()
	plt.figure(figsize=figsize(0.9,0.5))
	
	
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

	savefig(file)
	end = time.clock()
	print('Created figure \'%s\' in %f s' % (file, end-start))


def CFMMSD():
	start = time.clock()
	file = 'CFMMSD'
	datafile = 'CHCl3-sol-298K-npt-6-msd.xvg'
	data = pd.read_csv('Data/' + datafile, skiprows = 18, header = None, delim_whitespace = True)

	plt.clf()
	plt.figure(figsize=figsize(1,0.5))

	ax = plt.subplot(121)
	ax.plot(data[0], data[1], label = 'Simulation liquid')
	ax.set_xlabel('Time [$\\mathrm{ps}$]')
	ax.set_ylabel('MSD [$\\mathrm{nm^2}$]')
	ax.autoscale(False)
	
	fit = np.polyfit(data[0], data[1], 1)
	p2 = np.poly1d(fit)
	ax.plot(data[0], p2(data[0]),'r-', label = 'Linear fit')
	plt.legend(loc = 2,frameon = False)

	datafile = 'CHCl3-sol-298K-npt-heat-T508-msd.xvg'
	data = pd.read_csv('Data/' + datafile, skiprows = 18, header = None, delim_whitespace = True)

	ax = plt.subplot(122)
	ax.plot(data[0]/1000, data[1], label='Simulation gas')
	ax.set_xlabel('Time [$\\mathrm{ns}$]')
	ax.set_ylabel('MSD [$\\mathrm{nm^2}$]')
	ax.autoscale(False)
	
	fitdata = data[:][data[0]>=7500]
	fit = np.polyfit(fitdata[0], fitdata[1], 1)
	p1 = np.poly1d(fit)
	ax.plot(data[0]/1000, p1(data[0]),'r-', label = 'Linear fit')
	plt.legend(loc = 2,frameon = False)

	plt.tight_layout()

	savefig(file)
	end = time.clock()
	print('Created figure \'%s\' in %f s' % (file, end-start), ', diffusion coefficient liquid = %.4e' % (p2[1]/6), ', diffusion coefficient gas = %.4e' % (p1[1]/6))

def CFMRDF():
	start = time.clock()
	file = 'CFMRDF'
	datafile = 'CHCl3-sol-298K-npt-6-rdf.xvg'
	data = pd.read_csv('Data/' + datafile, skiprows = 18, header = None, delim_whitespace = True)

	plt.clf()
	plt.figure(figsize=figsize(1,0.5))

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

	savefig(file)
	end = time.clock()
	print('Created figure \'%s\' in %f s' % (file, end-start))


if __name__ == '__main__':
	print('Starting creating figures')
	start = time.clock()

	lock = Lock()
	queue = Queue()
	# pool = Pool()
	workers = 4
	processes = []

	# queue.put(MDSLEX2b)
	# queue.put(MDSLP1)
	# queue.put(MDSLP2)
	# queue.put(MDSLP3N4)
	# queue.put(MDSLP3N5)
	# queue.put(MDSLP3N6)
	# queue.put(MDSLP5N2)
	# queue.put(MDSLP5N3)
	# queue.put(MDSLP8N12)
	# queue.put(MDSLP8N24)
	queue.put(ArgonAnnealedDensTemp)
	queue.put(ArgonColdDensTemp)
	queue.put(ArgonMSD)
	queue.put(ArgonRDF)
	queue.put(CFMAnnealedDensTemp)
	queue.put(CFMColdDensTemp)
	queue.put(CFMMSD)
	queue.put(CFMRDF)

	for w in range(workers):
		p = Process(target = worker, args=(queue, lock))
		p.start()
		processes.append(p)
		queue.put('STOP')

	for p in processes:
		p.join()

	# while not queue.empty():
	# 	queue.get()()

	# p1[0] = Process(target = MDSLEX2b, args = ())

	# p1.start()
	# p1.join()

	end = time.clock()
	print('Finished in %f s' % (end-start))