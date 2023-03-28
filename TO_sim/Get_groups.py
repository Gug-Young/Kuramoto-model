from scipy.signal import find_peaks
import numpy as np
from TO_sim.Sol_Kuramoto import Sol_Kuramoto_mf2 as mf2
from TO_sim.gen_Distribution import Normal,Lorentzian,Quantile_Normal,Quantile_Lorentzian

mean = lambda x :np.mean(x[-500:])
get_std  = lambda x :np.std(x[-500:])

distribution = {
    'Normal':Normal,
    'Q_Normal':Quantile_Normal,
    'Lorentzian':Lorentzian,
    'Q_Lorentzian':Quantile_Lorentzian
}

def get_groups(dtheta,sum_time=500):
    dtheta_c = np.cumsum(dtheta,axis=0)
    avg_dtheta = (dtheta_c[sum_time:]-dtheta_c[:-sum_time])/sum_time
    num_data = []
    for data in avg_dtheta[-500:]:
        diff_dtheta = np.diff(data)
        peaks, _ = find_peaks(diff_dtheta, height=0.01)
        num_data.append(np.diff(peaks))
    groups = []
    N = len(dtheta[-1])
    for diff in num_data:
        sort_re = np.sort(diff)[::-1]
        temp = []
        for i in sort_re[:5]:
            temp.append(i)
        if len(sort_re) == 0:
            temp = [N,0,0,0,0]
        else:
            for i in range(5- len(sort_re)):
                temp.append(0)
        groups.append(temp)
    return groups

def make_meanr(K,m,N,theta_init,omega,t):
    theta, dtheta,rs = mf2(K,m=m,N=N,t_array=t,p_theta=theta_init,p_dtheta= omega,p_omega=omega)
    r = mean(rs)
    rstd = get_std(rs)
    g = np.mean(get_groups(dtheta)[-1000:],axis=0)
    g_std = np.std(get_groups(dtheta)[-1000:],axis=0)
    return r,rstd,g,g_std



class phase_diagram():
    def __init__(self,seed,N,m,t_end,dist='Normal') -> None:
        self.theta_init,omega,_ = distribution[dist](N,0,1,seed=seed)
        self.omega = np.sort(omega) 
        self.N = N
        self.t = np.arange(0,t_end+0.1/2,0.1)
        self.m = m

    def make_meanr(self,K,sum_time = 500):
        m = self.m
        t = self.t
        N = self.N
        theta_init = self.theta_init
        omega = self.omega
        theta, dtheta,rs = mf2(K,m=m,N=N,t_array=t,p_theta=theta_init,p_dtheta= omega,p_omega=omega)
        r = mean(rs)
        rstd = get_std(rs)
        groups = get_groups(dtheta,sum_time)[-sum_time:]
        g = np.mean(groups,axis=0)
        g_std = np.std(groups,axis=0)
        return r,rstd,g,g_std
