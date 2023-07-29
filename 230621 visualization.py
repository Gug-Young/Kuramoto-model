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
# df = pd.read_excel('N = 500 seed = 10 test dt = 0.1 230708 ver3.xlsm',index_col=[0,1])

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba 

def cuboid_data2(o, size=(1,1,1)):
    X = [[[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0]],
         [[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]],
         [[1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]],
         [[0, 0, 1], [0, 0, 0], [0, 1, 0], [0, 1, 1]],
         [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]],
         [[0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1]]]
    X = np.array(X).astype(float)
    for i in range(3):
        X[:,:,i] *= size[i]
    X += np.array(o)
    return X

def plotCubeAt2(positions,sizes=None,colors=None, **kwargs):
    if not isinstance(colors,(list,np.ndarray)): colors=["C0"]*len(positions)
    if not isinstance(sizes,(list,np.ndarray)): sizes=[(1,1,1)]*len(positions)
    g = []
    for p,s,c in zip(positions,sizes,colors):
        g.append( cuboid_data2(p, size=s) )
    return Poly3DCollection(np.concatenate(g),  
                            facecolors=np.repeat(colors,6), **kwargs)
    
df = pd.read_excel('N = 500 seed = 10 test dt = 0.1 230708 ver3.xlsm',index_col=[0,1])
# df.set_index(['K','m'])

# df = pd.read_excel('N = 500 seed = 10 test 230627 with initial phase vel.xlsm',index_col=[0,1])
# df = pd.read_excel('N = 500 seed = 10 test 230621.xlsm',index_col=[0,1])

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
Ss_c0 = df.reset_index()['c0'].to_numpy()
So_c0 = df.reset_index()['c0 omega'].fillna(0).to_numpy()
So_c1 = df.reset_index()['c1 omega'].fillna(0).to_numpy()
So_c2 = df.reset_index()['c2 omega'].fillna(0).to_numpy()
So_c3 = df.reset_index()['c3 omega'].fillna(0).to_numpy()
So_c4 = df.reset_index()['c4 omega'].fillna(0).to_numpy()
So_c5 = df.reset_index()['c5 omega'].fillna(0).to_numpy()

rMM = df.reset_index()['rMM']
rstd = df.reset_index()['rstd']


fig = plt.figure(figsize=(10,10))
plt.subplot(projection='3d')

ax = plt.gca()

positions = [(2,0,0.3),(2,0,-1.5)]
sizes = [(8,10,1.5), (8,10,1.4)]
colors = ["crimson","limegreen"]

# ax.set_aspect('equal') 
pc = plotCubeAt2(positions,sizes,colors=colors, edgecolor='#43ff6433',alpha=0.05)
ax.add_collection3d(pc)

ax.scatter(Ks,ms,So_c0,c=rstd,s=Ss_c0*0.05,zorder=10)
ax.scatter(Ks,ms,So_c1,c=rstd,s=Ss_c1*0.05,zorder=10)
ax.scatter(Ks,ms,So_c2,c=rstd,s=Ss_c2*0.05,zorder=10)
ax.scatter(Ks,ms,So_c3,c=rstd,s=Ss_c3*0.05,zorder=10)
ax.scatter(Ks,ms,So_c4,c=rstd,s=Ss_c4*0.05,zorder=10)
ax.scatter(Ks,ms,So_c5,c=rstd,s=Ss_c5*0.05,zorder=10)

ax.set_xlabel(r'$K$ : coupling constant')
ax.set_ylabel(r'$m$ : inertia')
ax.set_zlabel('cluster mean phase velocity')
# plt.zlim(-0.)
# plt.scatter(Ks,ms,Ss,Ss)

plt.tight_layout()
plt.show()

plt.show()



