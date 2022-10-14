from TO_sim.Hysteresis_Kuramoto import *
from TO_sim.Utility import *
from TO_sim.Utility import *
from TO_sim.To_Draw import *
def shuffle_seed_Hysteresis_pd(shuffle_seed,m,N=500,t_end=800,dist = "Quantile Lorentzian",dt = 0.1,dK=0.2): 
  
    Ksdf,Ksrdf = Hysteresis_pd_without_aram(m,N=N,dK=dK,t_end=t_end,dist=dist,dt=dt,seed="Uniform",shuffle=True,shuffle_seed=shuffle_seed)
    return Ksdf,Ksrdf,shuffle_seed