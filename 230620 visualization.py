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
df = pd.read_excel('test 230620.xlsm',index_col=[0,1])
# df_Km = df.reset_index(['K','m'])
KK = df.index.to_frame().K.unstack().to_numpy().T
MM = df.index.to_frame().m.unstack().to_numpy().T
# origin = 'lower'
# CS = plt.contourf(KK,MM,df['rMM'].unstack().T, 20, cmap=plt.cm.viridis, origin=origin)
# CS2 = plt.contour(CS, levels=CS.levels[::], colors='k',linewidths = 0.5, origin=origin)
# plt.colorbar()
Ks = df.reset_index().K
ms = df.reset_index().m
Ss_c0 = df.reset_index()['S c0']
Ss_c1 = df.reset_index()['S c1']
Ss_c2 = df.reset_index()['S c2']
Ss_c3 = df.reset_index()['S c3']
Ss_c4 = df.reset_index()['S c4']
Ss_c5 = df.reset_index()['S c5']

sm = np.zeros_like(Ss_c0)
for i in [Ss_c0,Ss_c1,Ss_c2,Ss_c3,Ss_c4,Ss_c5]:
    sm+=i.fillna(0)
# sm = Ss_c0 +Ss_c1+Ss_c2+Ss_c3+Ss_c4+Ss_c5


So_c0 = df.reset_index()['S c0 omega']
So_c1 = df.reset_index()['S c1 omega']
So_c2 = df.reset_index()['S c2 omega']
So_c3 = df.reset_index()['S c3 omega']
So_c4 = df.reset_index()['S c4 omega']
So_c5 = df.reset_index()['S c5 omega']


rMM = df.reset_index()['rMM']


# plt.subplot(projection='3d')
fig = plt.figure()
ax = fig.add_subplot(projection='3d')



# ax.scatter(Ks,ms,So_c0,c=rMM,s=Ss_c0*0.1)
# ax.scatter(Ks,ms,So_c1,c=rMM,s=Ss_c1*0.1)
# ax.scatter(Ks,ms,So_c2,c=rMM,s=Ss_c2*0.1)
# ax.scatter(Ks,ms,So_c3,c=rMM,s=Ss_c3*0.1)
# ax.scatter(Ks,ms,So_c4,c=rMM,s=Ss_c4*0.1)
# ax.scatter(Ks,ms,So_c5,c=rMM,s=Ss_c5*0.1)


ax.plot_surface(KK, MM, df['c0'], edgecolor='royalblue', lw=0.5, rstride=8, cstride=8,
                alpha=0.3)

# Plot projections of the contours for each dimension.  By choosing offsets
# that match the appropriate axes limits, the projected contours will sit on
# the 'walls' of the graph.
ax.contour(KK, MM, df['c0'], zdir='z', offset=-100, cmap='coolwarm')
ax.contour(KK, MM, df['c0'], zdir='x', offset=-40, cmap='coolwarm')
ax.contour(KK, MM, df['c0'], zdir='y', offset=40, cmap='coolwarm')

# ax.set(xlim=(-40, 40), ylim=(-40, 40), zlim=(-100, 100),
#        xlabel='X', ylabel='Y', zlabel='Z')

# ax.scatter(Ks,ms,sm)

# plt.zlim(-0.)
# plt.scatter(Ks,ms,Ss,Ss)
plt.show()


