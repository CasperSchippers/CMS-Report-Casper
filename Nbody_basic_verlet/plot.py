import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import pyplot as plt
import pandas as pd

# defining standard plotting functions


# def savefig(filename):
#     # plt.savefig('%s.pgf' % filename)
#     # plt.savefig('%s.pdf' % filename)
#     plt.savefig('%s.png' % filename)
#     # plt.savefig('%s.png' % filename)

# # mpl settings
# # mpl.use('pgf')

# pgf_with_latex = {
#     "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
#     "text.usetex": True,                # use LaTeX to write all text
#     "font.family": "serif",
#     "font.serif": [],                   # blank entries should cause plots to inherit fonts from the document
#     "font.sans-serif": [],
#     "font.monospace": [],
#     "axes.labelsize": 10,               # LaTeX default is 10pt font.
#     "font.size": 10,
#     "legend.fontsize": 8,               # Make the legend/label fonts a little smaller
#     "xtick.labelsize": 8,
#     "ytick.labelsize": 8,
#     "figure.figsize": figsize(0.9),     # default fig size of 0.9 textwidth
#     "pgf.preamble": [
#         r"\usepackage[utf8x]{inputenc}",    # use utf8 fonts becasue your computer can handle it :)
#         r"\usepackage[T1]{fontenc}",        # plots will be generated using this preamble
#         ],
#     "pgf.rcfonts": False                # use LaTeX to render the text with the rest of the LaTeX file
# }

# mpl.rcParams.update(pgf_with_latex)

# import pyplot after setting mpl
import matplotlib.pyplot as plt

# MDSLEX2b
# plt.figure(figsize=figsize(0.7, 0.6))

# x = np.arange(0.001, 3, 0.01)
# y = 4*((x**-12) - (x**-6))

# plt.plot(x, y)
# plt.gca().set_xlabel('$r/\sigma$')
# plt.gca().set_ylabel('$ U_\mathrm{LJ}/\\varepsilon$')

# plt.gcf().subplots_adjust(bottom=0.15)

# plt.ylim(-1.5, 5)


# savefig('MDSLEX2b')

# MDSLP1
data = pd.read_csv("trajectories", header=None, delim_whitespace=True)

# plt.figure()

N = round((np.shape(data)[1]-4)/6)
# print(len(data))

plt.figure(figsize=(15,7))

ax = plt.subplot(131, projection='3d')
for n in range(N):
	ax.plot(data[6*n+4], data[6*n+5], data[6*n+6],'o', markeredgewidth=0.0, markersize=1)
ax.view_init(20, 30)
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$z$')
# ax.tick_params(axis='z', pad=100)
# ax.xaxis.set_ticks(np.arange(0, 1.1, 0.5))
# ax.yaxis.set_ticks(np.arange(-0.5, 0.6, 0.5))
# ax.zaxis.set_ticks(np.arange(-0.1, 0.11, 0.1))
plt.tight_layout()

ax = plt.subplot(132, projection='3d')
for n in range(N):
	ax.plot(data[6*n+7], data[6*n+8], data[6*n+9],'o', markeredgewidth=0.0, markersize=1)
ax.view_init(20, 30)
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$z$')
# ax.tick_params(axis='z', pad=100)
# ax.xaxis.set_ticks(np.arange(0, 1.1, 0.5))
# ax.yaxis.set_ticks(np.arange(-0.5, 0.6, 0.5))
# ax.zaxis.set_ticks(np.arange(-0.1, 0.11, 0.1))
plt.tight_layout()

plt.subplot(133)
plt.plot(data[0],data[1], label='Kinetic energy')
plt.plot(data[0],data[2], label='Potential energy')
plt.plot(data[0],data[3], label='Total energy')
plt.gca().set_xlabel('Time $t$')
plt.gca().set_ylabel('Energy')


# plt.ylim(-1.05,0.8)
# plt.xlim(0,10)
plt.legend(bbox_to_anchor=(1, 0.8), frameon=False)
plt.tight_layout()

plt.savefig('trajectories.png')