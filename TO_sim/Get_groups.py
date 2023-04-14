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

def get_groups(dtheta,sum_time=1000):
    dtheta_c = np.cumsum(dtheta,axis=0)
    avg_dtheta = (dtheta_c[sum_time:]-dtheta_c[:-sum_time])/sum_time
    num_data = []
    for data in avg_dtheta[-1000:]:
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
<<<<<<< HEAD
    def __init__(self,seed,N,m,t_end,dist='Normal',Process='F',dt=0.1) -> None:
=======
    def __init__(self,seed,N,m,t_end,dt=0.1,dist='Normal',Process='F') -> None:
>>>>>>> main
        self.theta_init,omega,_ = distribution[dist](N,0,1,seed=seed)
        self.omega = np.sort(omega) 
        if Process == 'B':
            self.theta_init = np.ones_like(self.theta_init)
            self.dtheta_init = np.zeros_like(self.theta_init)
        else:
            self.dtheta_init = self.omega
        self.N = N
        self.t = np.arange(0,t_end+dt/2,dt)
        self.m = m
        self.sumtime = int(50/dt)

    def make_meanr(self,K,sum_time = 500,look_time = 0):
        m = self.m
        t = self.t
        N = self.N
        theta_init = self.theta_init
        dtheta_init = self.dtheta_init
        omega = self.omega
        theta, dtheta,rs = mf2(K,m=m,N=N,t_array=t,p_theta=theta_init,p_dtheta= dtheta_init,p_omega=omega,result_time=look_time)

        r = mean(rs)
        rstd = get_std(rs)
        groups = get_groups(dtheta,sum_time)[-sum_time:]
        g = np.mean(groups,axis=0)
        g_std = np.std(groups,axis=0)
        return r,rstd,g,g_std
    def hysteresis(self,m,Ks,sum_time = 500,look_time = 4000,way = 'F'):
        t = self.t
        N = self.N
        theta_init = self.theta_init
        dtheta_init = self.dtheta_init
        omega = self.omega
        K_r = []
        K_rstd = []
        K_groups = []
        K_g = []
        K_g_std = []
        len_K = len(Ks)
        for K in Ks:
            theta, dtheta,rs = mf2(K,m=m,N=N,t_array=t,p_theta=theta_init,p_dtheta= dtheta_init,p_omega=omega,result_time=look_time )
            theta_init = theta[-1]
            dtheta_init = dtheta[-1]
            K_r.append( mean(rs) )
            K_rstd.append( get_std(rs) )
            groups =  get_groups(dtheta,sum_time)[-sum_time:]
            K_g.append( np.mean(groups,axis=0) )
            K_g_std.append( np.std(groups,axis=0) )
        if way=='T':
            for K in Ks[::-1]:
                theta, dtheta,rs = mf2(K,m=m,N=N,t_array=t,p_theta=theta_init,p_dtheta= dtheta_init,p_omega=omega,result_time=look_time )
                theta_init = theta[-1]
                dtheta_init = dtheta[-1]
                K_r.append( mean(rs) )
                K_rstd.append( get_std(rs) )
                groups =  get_groups(dtheta,sum_time)[-sum_time:]
                K_g.append( np.mean(groups,axis=0) )
                K_g_std.append( np.std(groups,axis=0) )
            F_K_r = K_r[:len_K]
            F_K_rstd = K_rstd[:len_K]
            F_K_g = K_g[:len_K]
            F_K_g_std = K_g_std[:len_K]
            B_K_r = K_r[:len_K-1:-1]                
            B_K_rstd = K_rstd[:len_K-1:-1]                
            B_K_g = K_g[:len_K-1:-1]                
            B_K_g_std = K_g_std[:len_K-1:-1]                
            return (F_K_r,B_K_r),(F_K_rstd,B_K_rstd),(F_K_g,B_K_g),(F_K_g_std,B_K_g_std)
        return K_r,K_rstd,K_g,K_g_std
    
    def hysteresis_check(self,m,Ks,sum_time = 500,look_time = 4000,way = 'F'):
        t = self.t
        N = self.N
        theta_init = self.theta_init
        dtheta_init = self.dtheta_init
        omega = self.omega
        theta_dict = {}
        dtheta_dict = {}
        len_K = len(Ks)
        K_r = []
        K_rstd = []
        K_groups = []
        K_g = []
        K_g_std = []
        len_K = len(Ks)
        for K in Ks:
            theta, dtheta,rs = mf2(K,m=m,N=N,t_array=t,p_theta=theta_init,p_dtheta= dtheta_init,p_omega=omega,result_time=look_time )
            theta_init = theta[-1]
            dtheta_init = dtheta[-1]
            theta_dict[K] = theta
            dtheta_dict[K] = dtheta
            
        return theta_dict, dtheta_dict