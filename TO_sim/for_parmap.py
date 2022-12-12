from TO_sim.Hysteresis_Kuramoto import Hysteresis_pd_init_pvel_sim as Hp

N = 500
ss = 0
dK = 0.2
dt = 0.1

def initial_phasevel(seed,m,t_end,dist):
    df,rdf = Hp(m,N=N,t_end=t_end,dist = dist,dt = 0.1,dK=dK, shuffle = True, shuffle_seed = ss,Init_dtheta=True,Init_dtheta_seed=seed, seed=7)
    return df,rdf



def initial_phase(seed,m,t_end,dist):
    df,rdf = Hp(m,N=N,t_end=t_end,dist = dist,dt = 0.1,dK=dK, shuffle = True, shuffle_seed = seed,Init_dtheta=False,Init_dtheta_seed=0, seed=7)
    return df,rdf