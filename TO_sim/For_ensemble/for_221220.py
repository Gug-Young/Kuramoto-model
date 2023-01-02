from TO_sim.To_Draw import Draw_theoretical_wData as DD
from TO_sim.Hysteresis_Kuramoto import Hysteresis_pd_init_pvel as Hp 
from TO_sim.Hysteresis_Kuramoto import *
from TO_sim.gen_Distribution import *

N = 500
seed = 1
K = 0.1
t_end = 400
dt = 0.1
def get_tr(m,K,t_end=400,seed = 0,initial_speed= 0, dtheta_init = np.zeros(N)):
    theta_init, omega_init, Kc = Identical(N, initial_speed, seed=seed)
    _,_,_,rs,t = Sol_Kuramoto_mf(N,K,m,(0, t_end),dt=dt,
                    p_theta=theta_init,
                    p_dtheta=dtheta_init,
                    p_omega=omega_init,
                    distribution="Normal",
                    
                )
    return t,rs


def get_tr2(m,K,t_end=400, dtheta_init = np.zeros(N),init_dist= Identical(N, 0, seed=0),dt = 0.1):
    theta_init, omega_init, Kc = init_dist
    theta_s,dtheta_s,omega,rs,t = Sol_Kuramoto_mf(N,K,m,(0, t_end),dt=dt,
                    p_theta=theta_init,
                    p_dtheta=dtheta_init,
                    p_omega=omega_init,
                    distribution="Normal",
                    
                )
    return t,rs,(theta_s,dtheta_s,omega)