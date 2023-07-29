import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('cluster.csv')
K = df['K']
m = df['m']
rstd = df[r'$r$\'s temporal std']
O = df['cluster mean phase velocity']
S = df['cluster size']
fig = plt.figure(dpi=200)
ax = plt.subplot(111,projection='3d')
sca = ax.scatter(K,m,O,s=S*0.1,c=rstd,cmap='magma')
bar = plt.colorbar(sca,shrink=0.5)
ax.set_xlabel('$K$ : coupling constant')
ax.set_ylabel('$m$ : inertia')
ax.set_title('cluster mean phase velocity')
bar.set_label(r'$r$ temporal std.')
ax.view_init(elev=5., azim=-55, roll=0)
fig.tight_layout()
plt.show()
