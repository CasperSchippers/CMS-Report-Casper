import sys
import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import time
from multiprocessing import Pool, Process, Lock, Queue
import importlib
from subprocess import call
import os
import glob

def worker(q, tq, lock):
    # Get first figure to build from queue
    figurename = q.get()
    while not figurename == "STOP":
        # Build figure
        start = time.clock()
        module = importlib.import_module('plot_' + figurename)
        fig = module.plot()


        fig.savefig('%s.pgf' % figurename, dpi = 500)
        # fig.savefig('%s.pdf' % figurename, dpi = 2000)

        tq.put(figurename)

        # with cd("../"):
        # 	# print("pdflatex --jobname PGFFigures/" + figurename + " Report")
        #     call("pdflatex --jobname PGFFigures/" + figurename + " Report", stdout=open(os.devnull, 'wb'))

        end = time.clock()
        lock.acquire()
        try:
        	print("Created figure %s in %f s" % (figurename, end-start))
        finally:
        	lock.release()

        # Get new figure to build from queue
        figurename = q.get()

def texworker(tq, lock):
	with cd("../"):
		figurename = tq.get()
		while not figurename == "STOP":
			start = time.clock()
			call("pdflatex -interaction=batchmode --jobname PGFFigures/" + figurename + " Report")
			end = time.clock()
			lock.acquire()
			try:
				print("Build figure %s in %f s" % (figurename, end-start))
			finally:
				lock.release()
			figurename = tq.get()



class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)

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
    "pgf.preamble": [
        r"\usepackage[utf8x]{inputenc}",    # use utf8 fonts becasue your computer can handle it :)
        r"\usepackage[T1]{fontenc}",        # plots will be generated using this preamble
        ],
    "pgf.rcfonts": False                # use LaTeX to render the text with the rest of the LaTeX file
}

mpl.rcParams.update(pgf_with_latex)

# import pyplot after setting mpl
import matplotlib.pyplot as plt


if __name__ == '__main__':
    print('Starting creating figures')
    start = time.clock()

    lock = Lock()
    queue = Queue()
    texqueue = Queue()
    # pool = Pool()
    workers = 4
    processes = []

    queue.put("THEX2b")
    queue.put("MDSLEX2b")
    queue.put("MDSLP1")
    queue.put("MDSLP2")
    queue.put("MDSLP3N4")
    queue.put("MDSLP3N5")
    queue.put("MDSLP3N6")
    queue.put("MDSLP5N2")
    queue.put("MDSLP5N3")
    queue.put("MDSLP8Susp")
    queue.put("MDSLP9N12")
    queue.put("MDSLP9N24")
    queue.put("ArgonAnnealedDensTemp")
    queue.put("ArgonColdDensTemp")
    queue.put("ArgonMSD")
    queue.put("ArgonRDF")
    queue.put("CFMAnnealedDensTemp")
    queue.put("CFMColdDensTemp")
    queue.put("CFMMSD")
    queue.put("CFMRDF")

    for w in range(workers):
        p = Process(target = worker, args=(queue, texqueue, lock))
        p.start()
        processes.append(p)
        queue.put('STOP')

    texp = Process(target = texworker, args=(texqueue, lock))
    texp.start()

    for p in processes:
        p.join()

    texqueue.put("STOP")
    texp.join()

    # while not queue.empty():
    #   queue.get()()

    # p1[0] = Process(target = MDSLEX2b, args = ())

    # p1.start()
    # p1.join()



    end = time.clock()
    print('Finished in %f s' % (end-start))