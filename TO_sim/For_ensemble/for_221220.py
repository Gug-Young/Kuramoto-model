from TO_sim.To_Draw import Draw_theoretical_wData as DD
from TO_sim.Hysteresis_Kuramoto import Hysteresis_pd_init_pvel as Hp 
from TO_sim.Hysteresis_Kuramoto import *
from TO_sim.gen_Distribution import *

N = 500
seed = 1
K = 0.1
t_end = 400
dt = 0.1
def get_tr(m,K,t_end=400,seed = 0,initial_speed= 0):
    theta_init, omega_init, Kc = Identical(N, initial_speed, seed=seed)
    dtheta_init = 0*np.random.random(N)
    _,_,_, rs, t = Sol_Kuramoto_mf(N,K,m,(0, t_end),dt=dt,
                    p_theta=theta_init,
                    p_dtheta=dtheta_init,
                    p_omega=omega_init,
                    distribution="Normal",
                    
                )
    return t,rs