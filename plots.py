import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

mean_values = pd.read_csv('result.csv').groupby(['NumThreads','Dim'], as_index = False).mean()

fig = plt.figure()

rt_subplt = fig.add_subplot(212)
Sp_subplt = fig.add_subplot(221)
Ep_subplt = fig.add_subplot(222)

rt_subplt.set_title('runtime')
rt_subplt.set_xlabel('threads')
rt_subplt.set_ylabel('time')
Sp_subplt.set_title('Speedup')
Sp_subplt.set_xlabel('threads')
Sp_subplt.set_ylabel("$S_{p}$")
Ep_subplt.set_title('Efficiency')
Ep_subplt.set_xlabel('threads')
Ep_subplt.set_ylabel("$E_{p}$")

for dimension in sorted(set(mean_values['Dim'])):
	sub_df = mean_values.loc[mean_values.Dim == dimension]
	one_thread_t = float(sub_df[sub_df.NumThreads == 1]['Runtime'])
	speedup = one_thread_t / np.array(sub_df['Runtime'])
	efficiency = speedup / np.array(sub_df['NumThreads']) 

	rt_subplt.plot(sub_df['NumThreads'], sub_df['Runtime'], marker = "o", label = "{dim}x{dim}".format(dim = dimension))
	Sp_subplt.plot(sub_df['NumThreads'], speedup, marker=".", label="{dim}x{dim}".format(dim = dimension))
	Ep_subplt.plot(sub_df['NumThreads'], efficiency, marker=".", label="{dim}x{dim}".format(dim = dimension))
rt_subplt.legend()

plt.subplots_adjust(wspace=0.5, hspace=0.6)
plt.show()