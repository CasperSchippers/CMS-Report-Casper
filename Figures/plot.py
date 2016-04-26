import numpy as np
import matplotlib as mpl

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

# def newfig():
    

def savefig(filename):
    plt.savefig('%s.pgf' % filename)
    plt.savefig('%s.pdf' % filename)
    plt.savefig('%s.png' % filename)

# mpl settings
mpl.use('pgf')

pgf_with_latex = {
    "figure.figsize": figsize(0.9),
    "pgf.rcfonts": False
}

mpl.rcParams.update(pgf_with_latex)

# import pyplot after setting mpl
import matplotlib.pyplot as plt