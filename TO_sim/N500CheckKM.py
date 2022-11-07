from TO_sim.Hysteresis_Kuramoto import *
from TO_sim.gen_Distribution import Quantile_Lorentzian
from TO_sim.Utility import  *
from TO_sim.Check_theorical import *
import numpy as np
N = 500
seed = 'uniform shuffle'
seed_shuffle = 7
theta_init, omega_init, Kc = Quantile_Lorentzian(N, 0, 1, seed=seed)
seeds = 2
# _, omega_init, Kc = Lorentzian(N, 0, 1, seed=seeds)
_, omega_init, Kc = Quantile_Lorentzian(N, 0, 1, seed=seeds)
ones = np.ones(N)
dtheta_init = np.zeros(N)
np.random.seed(seed_shuffle)
np.random.shuffle(theta_init)
t_end = 800
dt = 0.1

def Check_Km_F(K,m):
    _,_,_, rs, t = Sol_Kuramoto_mf(N,K,m,(0, t_end),dt=dt,
                p_theta=theta_init,
                p_dtheta=dtheta_init,
                p_omega=omega_init,
                distribution="Quantile Lorentzian",)
    return t, rs

def check_case1_KR(m):
    Ks = np.linspace(0.1,6,1000)
    KF,RF,KB,RB = Make_theorical_KR(Ks,m)
    min_KF = min(KF)
    min_RF = min(RF)
    return min_KF,min_RF

def check_case2_KR(m):
    Ks = np.linspace(0.1,6,1000)
    KF,RF,KB,RB = Make_theorical_KR(Ks,m)
    min_KBidx = np.argmin(KB)
    min_KB = KB[min_KBidx]
    min_RB = RB[min_KBidx]
    return min_KB,min_RB

def Check_Km_B(K,m):
    _,_,_, rs, t = Sol_Kuramoto_mf(N,K,m,(0, t_end),dt=dt,
                p_theta=ones,
                p_dtheta=dtheta_init,
                p_omega=omega_init,
                distribution="Quantile Lorentzian",)
    return t, rs

def Draw_case11_KR(m):
    Ks = np.linspace(1,12.5,1000)
    KF,RF,KB,RB = Make_theorical_KR(Ks,m)
    return KF,RF,KB,RB