import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from TO_sim.Sol_Kuramoto import Sol_Kuramoto_mK as mK
from TO_sim.Sol_Kuramoto import Sol_Kuramoto_mf2 as mf2
from TO_sim.get_cluster import C_rsmso,cluster_os
from TO_sim.To_Draw import Draw_avg_vel_r

from TO_sim.gen_Distribution import Normal
from TO_sim.gen_Distribution import Quantile_Normal as Q_Normal
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from tqdm.notebook import tqdm
from scipy.signal import find_peaks
import parmap
# df = pd.read_excel('N = 500 seed = 10 test 230627.xlsm',index_col=[0,1])
df = pd.read_excel('N = 500 seed = 10 test 230627 with initial phase vel.xlsm',index_col=[0,1])
# df_Km = df.reset_index(['K','m'])
KK = df.index.to_frame().K.unstack().to_numpy().T
MM = df.index.to_frame().m.unstack().to_numpy().T
# origin = 'lower'
# CS = plt.contourf(KK,MM,df['rMM'].unstack().T, 20, cmap=plt.cm.viridis, origin=origin)
# CS2 = plt.contour(CS, levels=CS.levels[::], colors='k',linewidths = 0.5, origin=origin)
# plt.colorbar()
Ks = df.reset_index().K
ms = df.reset_index().m
Ss_c0 = df.reset_index()['c0'].to_numpy()
Ss_c1 = df.reset_index()['c1'].to_numpy()
Ss_c2 = df.reset_index()['c2'].to_numpy()
Ss_c3 = df.reset_index()['c3'].to_numpy()
Ss_c4 = df.reset_index()['c4'].to_numpy()
Ss_c5 = df.reset_index()['c5'].to_numpy()
So_c0 = df.reset_index()['c0 omega'].fillna(0).to_numpy()
So_c1 = df.reset_index()['c1 omega'].fillna(0).to_numpy()
So_c2 = df.reset_index()['c2 omega'].fillna(0).to_numpy()
So_c3 = df.reset_index()['c3 omega'].fillna(0).to_numpy()
So_c4 = df.reset_index()['c4 omega'].fillna(0).to_numpy()
So_c5 = df.reset_index()['c5 omega'].fillna(0).to_numpy()

rMM = df.reset_index()['rMM']


fig = plt.figure(figsize=(10,10))
plt.subplot(projection='3d')
ax = plt.gca()
ax.scatter(Ks,ms,So_c0,c=rMM,s=Ss_c0*0.01)
ax.scatter(Ks,ms,So_c1,c=rMM,s=Ss_c1*0.01)
ax.scatter(Ks,ms,So_c2,c=rMM,s=Ss_c2*0.01)
ax.scatter(Ks,ms,So_c3,c=rMM,s=Ss_c3*0.01)
ax.scatter(Ks,ms,So_c4,c=rMM,s=Ss_c4*0.01)
ax.scatter(Ks,ms,So_c5,c=rMM,s=Ss_c5*0.01)

ax.set_xlabel(r'$K$ : coupling constant')
ax.set_ylabel(r'$m$ : inertia')
ax.set_zlabel('cluster phase velocity')
# plt.zlim(-0.)
# plt.scatter(Ks,ms,Ss,Ss)
plt.show()

