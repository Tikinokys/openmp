#! /usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def build_efficiency_plots(mean_values, file):
	# построение графиков эффективности, ускорения и вермени работы 
	fig = plt.figure()
	rt_subplt = fig.add_subplot(212)
	Sp_subplt = fig.add_subplot(221)
	Ep_subplt = fig.add_subplot(222)

	rt_subplt.set_title('runtime'); rt_subplt.set_xlabel('threads'); rt_subplt.set_ylabel('time')
	Sp_subplt.set_title('Speedup'); Sp_subplt.set_xlabel('threads'); Sp_subplt.set_ylabel("$S_{p}$")
	Ep_subplt.set_title('Efficiency'); Ep_subplt.set_xlabel('threads'); Ep_subplt.set_ylabel("$E_{p}$")

	for dimension in sorted(set(mean_values['Dim'])):
		sub_df = mean_values.loc[mean_values.Dim == dimension]
		one_thread_t = sub_df[sub_df.NumThreads == 1]['Runtime'].astype(float)
		speedup = np.array(one_thread_t) / np.array(sub_df['Runtime'])
		efficiency = speedup / np.array(sub_df['NumThreads']) 

		rt_subplt.plot(sub_df['NumThreads'], sub_df['Runtime'], marker = "o", label = "{dim}x{dim}".format(dim = dimension))
		Sp_subplt.plot(sub_df['NumThreads'], speedup, marker=".", label="{dim}x{dim}".format(dim = dimension))
		Ep_subplt.plot(sub_df['NumThreads'], efficiency, marker=".", label="{dim}x{dim}".format(dim = dimension))
	rt_subplt.legend()
	plt.subplots_adjust(wspace=0.5, hspace=0.6)
	plt.savefig('{}.pdf'.format(file))


def build_surface_plot(surface, details, file):
	# построение графика поверхности
	X = np.arange(0, 1, 1.0/surface.shape[0])
	Y = np.arange(0, 1, 1.0/surface.shape[1])

	X2D,Y2D = np.meshgrid(X, Y)
	fig = plt.figure()
	ax = Axes3D(fig)
	ax.plot_surface(X2D,Y2D, surface)
	plt.suptitle("threads: {}, iterations: {}, runtime: {}s, dim: {}, eps: {}".format(details[0], details[1], details[2], details[3], details[4]))
	# plt.savefig("{}.pdf".format(file))
	plt.show()

details = pd.read_csv("dirichlet_benchmark.csv").groupby(['NumThreads','Dim'], as_index = False).mean()
build_efficiency_plots(details, "d_plots")

df = pd.read_csv("surface.csv", header=None, sep=" ")
build_surface_plot(np.array(df[:-1].astype(float)), df.tail(1)[0].values[0].split(","), "d_surface")