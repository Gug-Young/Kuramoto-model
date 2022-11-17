from TO_sim.Sol_Kuramoto import Sol_Kuramoto_mf_r_t as SOL_forward
from TO_sim.Sol_Kuramoto import Sol_Kuramoto_mf_rb_t as SOL_back
from TO_sim.Hysteresis_Kuramoto import Hysteresis_pd_sim as Hys_sim
import numpy as np

t_end = 800
t_end_HY = 800
N = 500 
dist = 'Lorentzian'
dt = 0.1
t = np.arange(0,t_end+dt/2,dt)
def sim_forward(seed,K,m):
    r,t = SOL_forward(N,K,m,tspan=(0,t_end),dt=0.1,seed=seed,distribution='Lorentzian')
    return r

def sim_backward(seed,K,m):
    r,t = SOL_back(N,K,m,tspan=(0,t_end),dt=0.1,seed=seed,distribution='Lorentzian')
    return r

def sim_Hystersis(seed,m,dK=0.2):
    Ksdf,Ksrdf = Hys_sim(m,dist='Lorentzian',seed=seed,t_end = t_end_HY,dK=dK)
    return Ksdf,Ksrdf
    