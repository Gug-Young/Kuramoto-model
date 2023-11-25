import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from TO_sim.Sol_Kuramoto import Sol_Kuramoto_mf2 as mf2
from TO_sim.Kuramoto_model import Kuramoto_2nd_mf,Kuramoto_1st_mf
from scipy.stats import norm
from TO_sim.get_cluster import cluster_os_new2
from TO_sim.gen_Distribution import Normal, Quantile_Normal as Q_Normal, Lorentzian
import TO_sim.analytical.order_sec_parameter as OSP
import TO_sim.analytical.sec_order_parameter2 as OSP2
import TO_sim.Integrator_jit as IJ

RK4_jit = IJ.RK4
RK4_jit_short = IJ.RK4_short


class Q_Norm_simul():
    def __init__(self,N,K,m,dt = 0.1,t_end = 5000,seed = 10,start_p = False) -> None:
        '''input : N,K,m,dt. dt, t_end,seed'''
        self.seed = seed
        theta_random, omega, Kc = Q_Normal(N, 0, 1, seed=seed)
        omega = np.sort(omega)-np.mean(omega)
        self.N = N; self.K = K; self.m = m;self.dt = dt
        self.t_end = t_end; self.dt = dt
        self.omega = omega
        self.t = np.arange(0, self.t_end+self.dt/2, self.dt)
        Theta = np.zeros(2*N)
        Theta[:N] = theta_random 
        if start_p:
            Theta[N:2*N] = omega
        self.Theta_init = Theta
        self.Theta_ori = Theta
    def solve(self):
        t = self.t
        if self.m == 0:
            func = Kuramoto_1st_mf
        else:
            func = Kuramoto_2nd_mf
            
        sol = RK4_jit(func,self.Theta_init,t, args=(self.omega, self.N, self.m, self.K))
        self.Last_sol = sol[-1]
        N = self.N
        theta,dtheta = sol[:,:N],sol[:,N:2*N]
        rs = np.abs(np.mean(np.exp(theta.T*1j),axis=0))
        if self.dt <0.1:
            t = t[::10]
            theta = theta[::10]
            dtheta = dtheta[::10]
            rs = rs[::10]
        self.rs = rs
        self.theta = theta
        self.dtheta = dtheta
        self.Theta_last = sol[-1]
        self.r_mean = np.mean(rs[-500:])
        solution = {}
        solution['rs'] = rs
        solution['r_mean'] = self.r_mean
        solution['r_std'] = np.std(rs[-2000:])
        solution['t'] = t
        solution['theta'] = theta
        solution['dtheta'] = dtheta
        solution['Theta_last'] = sol[-1]
        return solution
    
    def solve_short(self,result_time=2010):
        t = self.t
        if self.m == 0:
            func = Kuramoto_1st_mf
        else:
            func = Kuramoto_2nd_mf
            
        sol = RK4_jit_short(func,self.Theta_init,t, args=(self.omega, self.N, self.m, self.K),result_time=result_time)
        self.Last_sol = sol[-1]
        N = self.N
        theta,dtheta = sol[:,:N],sol[:,N:2*N]
        rs = np.abs(np.mean(np.exp(theta.T*1j),axis=0))
        if self.dt <0.1:
            t = t[::10]
            theta = theta[::10]
            dtheta = dtheta[::10]
            rs = rs[::10]
        self.rs = rs
        self.theta = theta
        self.dtheta = dtheta
        self.Theta_last = sol[-1]
        self.r_mean = np.mean(rs[-500:])
        solution = {}
        solution['rs'] = rs
        solution['r_mean'] = self.r_mean
        solution['r_std'] = np.std(rs[-2000:])
        solution['t'] = t
        solution['theta'] = theta
        solution['dtheta'] = dtheta
        solution['Theta_last'] = sol[-1]
        return solution
    
    def get_cluster(self,sum_time=2000):
        N = self.N
        avg_dtheta = np.array([np.mean(self.dtheta[-sum_time-i:-i],axis=0) for i in range(10,0,-1)])
        c_threshold = np.where(self.r_mean<0.05,1e-5,2e-3)
        CS, CMP, cluster, omega_s, omega_e, CMO, Is_group, C_s, C_e = cluster_os_new2(
            avg_dtheta[-10:], height=15e-2, c_std=3, check=c_threshold, c_size=3, N=N, omega=self.omega)
        CS,CMP
        if len(CS) == 0:
            sp0 = np.nan
        else:
           sp0 = CMP[0]
        names = {}
        colors = {}
        CS_s = {}
        CS_p = {}
        CS_o = {}
        cluster_s = {}
        num = 0
        cluster_info = {}
        P_idx,=np.where(self.omega>0)
        M_idx,=np.where(self.omega<0)
        for clu in cluster:
            P_idx = np.setdiff1d(P_idx, clu)
            M_idx = np.setdiff1d(M_idx, clu)
            break

        for size,sp,clu in zip(CS,CMP,cluster):
            ro = sp - sp0
            if num >=3: break
            if ro==0:
                cname='0';names[cname]='Main cluster';
                colors[cname]='gold';
                CS_s[cname] = size;cluster_s[cname]=clu
                CS_p[cname] = sp 
            elif ro<0:
                cname='-';names[cname]=r'secondary cluster$(-)$'
                colors[cname]='aqua';
                CS_s[cname] = size;cluster_s[cname]=clu
                CS_p[cname] = sp
            else:
                cname='+';names[cname]=r'secondary cluster$(+)$'
                colors[cname]='fuchsia';
                CS_s[cname] = size;cluster_s[cname]=clu
                CS_p[cname] = sp
            num+=1
        CS_s['+_total'] = len(P_idx);cluster_s['+_total']=P_idx
        CS_p['+_total'] = np.mean(avg_dtheta[-1][P_idx])
        CS_s['-_total'] = len(M_idx);cluster_s['-_total']=M_idx
        CS_p['-_total'] = np.mean(avg_dtheta[-1][M_idx])

        cluster_info['c_name'] = names
        cluster_info['c_size'] = CS_s
        cluster_info['c_speed'] = CS_p
        cluster_info['c_color'] = colors
        cluster_info['c_cluster'] = cluster_s
        cluster_info['avg_dtheta_last'] = avg_dtheta[-1]
        self.cluster_info = cluster_info
        return cluster_info
    def get_r_clu(self,sum_time=500):
        N = self.N
        r_clu_info = {}
        cluster_info = self.get_cluster()
        cluster_s = cluster_info['c_cluster']
        c_name = cluster_s.keys()
        r_clus = {}
        r_clus_mean = {}
        r_clus_std = {}
        r_clus_mean_last = {}
        psi_clu = {}
        rs = self.rs
        for name  in c_name:
            clu = cluster_s[name]
            temp = 1/N*np.sum(np.exp(1j*self.theta[:, clu]), axis=1)
            rc = np.abs(temp)
            rc_mean = np.array([np.mean(rc[i:i+sum_time], axis=0) for i in range(len(self.t)-sum_time)])
            rc_std = np.array([np.std(rc[i:i+sum_time], axis=0) for i in range(len(self.t)-sum_time)])
            psic = np.angle(temp)
            r_clus[name]=rc
            r_clus_mean[name]=rc_mean
            r_clus_std[name]=rc_std
            r_clus_mean_last[name]=rc_mean[-1]
            psi_clu[name]=psic
        rs_mean = np.array([np.mean(rs[i:i+sum_time], axis=0) for i in range(len(self.t)-sum_time)])
        rs_std = np.array([np.std(rs[i:i+sum_time], axis=0) for i in range(len(self.t)-sum_time)])
        r_clu_info['t'] =  self.t
        r_clu_info['t_mean'] =  self.t[sum_time:]
        r_clu_info['clu_name'] =  c_name
        r_clu_info['r_clu'] =  r_clus
        r_clu_info['r_clu_mean'] =  r_clus_mean
        r_clu_info['r_clu_mean_last'] =  r_clus_mean_last
        r_clu_info['r_clu_std'] =  r_clus_std
        r_clu_info['psi_clu'] =  psi_clu
        r_clu_info['r_total_mean'] =  rs_mean
        r_clu_info['r_total_std'] =  rs_std
        return r_clu_info
    
    
    def get_r_clu_last(self,sum_time=500):
        N = self.N
        r_clu_info = {}
        cluster_info = self.get_cluster()
        cluster_s = cluster_info['c_cluster']
        c_name = cluster_s.keys()
        r_clus_std = {}
        r_clus_mean_last = {}
        rs = self.rs
        for name  in c_name:
            clu = cluster_s[name]
            temp = 1/N*np.sum(np.exp(1j*self.theta[-sum_time:, clu]), axis=1)
            rc = np.abs(temp)
            rc_mean = np.array([np.mean(rc[:sum_time], axis=0)])
            rc_std = np.array([np.std(rc[:sum_time], axis=0)])
            r_clus_std[name]=rc_std
            r_clus_mean_last[name]=rc_mean[-1]
        rs_mean = np.array([np.mean(rs[:sum_time], axis=0)])
        rs_std = np.array([np.std(rs[:sum_time], axis=0)])
        r_clu_info['clu_name'] =  c_name
        r_clu_info['r_clu_mean_last'] =  r_clus_mean_last
        r_clu_info['r_clu_std'] =  r_clus_std
        r_clu_info['r_total_mean'] =  rs_mean
        r_clu_info['r_total_std'] =  rs_std
        return r_clu_info
    def TLO(self,K_end = 15,dK = 0.1):
        N = self.N
        self.dK = dK
        Ks = np.arange(0,K_end+dK/2,dK)
        self.Theta_last = self.Theta_ori.copy()
        df_rset = pd.DataFrame(columns=['r_mean','r0','r+','r-','r+_total','r-_total',
                                        'sig_mean','sig0','sig+','sig-','sig+_total','sig-_total'],index=Ks)
        df_cluster = pd.DataFrame(columns=['S0','S+','S-','S+_total','S-_total',
                                           'v0','v+','v-','v+_total','v-_total',
                                           'max_O0','max_O+','max_O-','max_O+_total','max_O-_total',
                                           'min_O0','min_O+','min_O-','min_O+_total','min_O-_total',
                                           'mean_O0','mean_O+','mean_O-','mean_O+_total','mean_O-_total'],index=Ks)
        df_cluster_idx = pd.DataFrame(columns=['CLU0','CLU+','CLU-','CLU+_total','CLU-_total'],index=Ks)
        df_avglast = pd.DataFrame(columns=range(N),index=Ks)
        df_Thetalast = pd.DataFrame(columns=range(2*N),index=Ks)

        for K in Ks:
            self.K = K
            self.Theta_init = self.Theta_last
            sol = self.solve()
            clu_info = self.get_cluster()
            r_info = self.get_r_clu()
            # print(r_info)
            c_type = r_info['clu_name']
            r_cl = r_info['r_clu_mean_last']
            sig_c = r_info['r_clu_std']
            df_rset.loc[K]['r_mean'] = r_info['r_total_mean'][-1]
            df_rset.loc[K]['sig_mean'] = r_info['r_total_std'][-1]
            for c_t in c_type:
                if c_t == '0':
                    clu = clu_info['c_cluster'][c_t]
                    df_rset.loc[K]['r0'] = r_cl[c_t]
                    df_rset.loc[K]['sig0'] = sig_c[c_t][-1]
                    df_cluster.loc[K]['S0'] = clu_info['c_size'][c_t]
                    df_cluster.loc[K]['v0'] = clu_info['c_speed'][c_t]
                    df_cluster.loc[K]['max_O0'] = np.max(self.omega[clu])
                    df_cluster.loc[K]['min_O0'] = np.min(self.omega[clu])
                    df_cluster.loc[K]['mean_O0'] = np.mean(self.omega[clu])
                    df_cluster_idx.loc[K]['CLU0'] = np.sort(clu)
                if c_t == '+':
                    clu = clu_info['c_cluster'][c_t]
                    df_rset.loc[K]['r+'] = r_cl[c_t]
                    df_rset.loc[K]['sig+'] = sig_c[c_t][-1]
                    df_cluster.loc[K]['S+'] = clu_info['c_size'][c_t]
                    df_cluster.loc[K]['v+'] = clu_info['c_speed'][c_t]
                    df_cluster.loc[K]['max_O+'] = np.max(self.omega[clu])
                    df_cluster.loc[K]['min_O+'] = np.min(self.omega[clu])
                    df_cluster.loc[K]['mean_O+'] = np.mean(self.omega[clu])
                    df_cluster_idx.loc[K]['CLU+'] = np.sort(clu)
                if c_t == '-':
                    clu = clu_info['c_cluster'][c_t]
                    df_rset.loc[K]['r-'] = r_cl[c_t]
                    df_rset.loc[K]['sig-'] = sig_c[c_t][-1]
                    df_cluster.loc[K]['S-'] = clu_info['c_size'][c_t]
                    df_cluster.loc[K]['v-'] = clu_info['c_speed'][c_t]
                    df_cluster.loc[K]['max_O-'] = np.max(self.omega[clu])
                    df_cluster.loc[K]['min_O-'] = np.min(self.omega[clu])
                    df_cluster.loc[K]['mean_O-'] = np.mean(self.omega[clu])
                    df_cluster_idx.loc[K]['CLU-'] = np.sort(clu)
                if c_t == '+_total':
                    clu = clu_info['c_cluster'][c_t]
                    df_rset.loc[K]['r+_total'] = r_cl[c_t]
                    df_rset.loc[K]['sig+_total'] = sig_c[c_t][-1]
                    df_cluster.loc[K]['S+_total'] = clu_info['c_size'][c_t]
                    df_cluster.loc[K]['v+_total'] = clu_info['c_speed'][c_t]
                    df_cluster.loc[K]['max_O+_total'] = np.max(self.omega[clu])
                    df_cluster.loc[K]['min_O+_total'] = np.min(self.omega[clu])
                    df_cluster.loc[K]['mean_O+_total'] = np.mean(self.omega[clu])
                    df_cluster_idx.loc[K]['CLU+_total'] = np.sort(clu)
                if c_t == '-_total':
                    clu = clu_info['c_cluster'][c_t]
                    df_rset.loc[K]['r-_total'] = r_cl[c_t]
                    df_rset.loc[K]['sig-_total'] = sig_c[c_t][-1]
                    df_cluster.loc[K]['S-_total'] = clu_info['c_size'][c_t]
                    df_cluster.loc[K]['v-_total'] = clu_info['c_speed'][c_t]
                    df_cluster.loc[K]['max_O-_total'] = np.max(self.omega[clu])
                    df_cluster.loc[K]['min_O-_total'] = np.min(self.omega[clu])
                    df_cluster.loc[K]['mean_O-_total'] = np.mean(self.omega[clu])
                    df_cluster_idx.loc[K]['CLU-_total'] = np.sort(clu)
            df_avglast.loc[K] = clu_info['avg_dtheta_last']
            df_Thetalast.loc[K] = self.Theta_last
        TLO_info = {}
        TLO_info['Ks'] = Ks
        TLO_info['r_info'] = df_rset
        TLO_info['cluster_info'] = df_cluster 
        TLO_info['avg_dtheta'] = df_avglast
        TLO_info['Theta_last'] = df_Thetalast
        TLO_info['CLU_idx'] = df_cluster_idx
        return TLO_info
    
    def TLO_back(self,Theta,K_back = 15,dK = 0.1):
        N = self.N
        self.dK = dK

        Ks = np.arange(0,K_back+dK/2,dK)
        self.Theta_last = Theta
        df_rset = pd.DataFrame(columns=['r_mean','r0','r+','r-','r+_total','r-_total',
                                        'sig_mean','sig0','sig+','sig-','sig+_total','sig-_total'],index=Ks)
        df_cluster = pd.DataFrame(columns=['S0','S+','S-','S+_total','S-_total',
                                           'v0','v+','v-','v+_total','v-_total',
                                           'max_O0','max_O+','max_O-','max_O+_total','max_O-_total',
                                           'min_O0','min_O+','min_O-','min_O+_total','min_O-_total',
                                           'mean_O0','mean_O+','mean_O-','mean_O+_total','mean_O-_total'],index=Ks)
        df_cluster_idx = pd.DataFrame(columns=['CLU0','CLU+','CLU-','CLU+_total','CLU-_total'],index=Ks)
        df_avglast = pd.DataFrame(columns=range(N),index=Ks)
        df_Thetalast = pd.DataFrame(columns=range(2*N),index=Ks)

        for K in Ks[::-1]:
            self.K = K
            self.Theta_init = self.Theta_last
            sol = self.solve()
            clu_info = self.get_cluster()
            r_info = self.get_r_clu_last()
            c_type = r_info['clu_name']
            r_cl = r_info['r_clu_mean_last']
            sig_c = r_info['r_clu_std']
            df_rset.loc[K]['r_mean'] = r_info['r_total_mean'][-1]
            df_rset.loc[K]['sig_mean'] = r_info['r_total_std'][-1]
            for c_t in c_type:
                if c_t == '0':
                    clu = clu_info['c_cluster'][c_t]
                    df_rset.loc[K]['r0'] = r_cl[c_t]
                    df_rset.loc[K]['sig0'] = sig_c[c_t][-1]
                    df_cluster.loc[K]['S0'] = clu_info['c_size'][c_t]
                    df_cluster.loc[K]['v0'] = clu_info['c_speed'][c_t]
                    df_cluster.loc[K]['max_O0'] = np.max(self.omega[clu])
                    df_cluster.loc[K]['min_O0'] = np.min(self.omega[clu])
                    df_cluster.loc[K]['mean_O0'] = np.mean(self.omega[clu])
                    df_cluster_idx.loc[K]['CLU0'] = np.sort(clu)
                if c_t == '+':
                    clu = clu_info['c_cluster'][c_t]
                    df_rset.loc[K]['r+'] = r_cl[c_t]
                    df_rset.loc[K]['sig+'] = sig_c[c_t][-1]
                    df_cluster.loc[K]['S+'] = clu_info['c_size'][c_t]
                    df_cluster.loc[K]['v+'] = clu_info['c_speed'][c_t]
                    df_cluster.loc[K]['max_O+'] = np.max(self.omega[clu])
                    df_cluster.loc[K]['min_O+'] = np.min(self.omega[clu])
                    df_cluster.loc[K]['mean_O+'] = np.mean(self.omega[clu])
                    df_cluster_idx.loc[K]['CLU+'] = np.sort(clu)
                if c_t == '-':
                    clu = clu_info['c_cluster'][c_t]
                    df_rset.loc[K]['r-'] = r_cl[c_t]
                    df_rset.loc[K]['sig-'] = sig_c[c_t][-1]
                    df_cluster.loc[K]['S-'] = clu_info['c_size'][c_t]
                    df_cluster.loc[K]['v-'] = clu_info['c_speed'][c_t]
                    df_cluster.loc[K]['max_O-'] = np.max(self.omega[clu])
                    df_cluster.loc[K]['min_O-'] = np.min(self.omega[clu])
                    df_cluster.loc[K]['mean_O-'] = np.mean(self.omega[clu])
                    df_cluster_idx.loc[K]['CLU-'] = np.sort(clu)
                if c_t == '+_total':
                    clu = clu_info['c_cluster'][c_t]
                    df_rset.loc[K]['r+_total'] = r_cl[c_t]
                    df_rset.loc[K]['sig+_total'] = sig_c[c_t][-1]
                    df_cluster.loc[K]['S+_total'] = clu_info['c_size'][c_t]
                    df_cluster.loc[K]['v+_total'] = clu_info['c_speed'][c_t]
                    df_cluster.loc[K]['max_O+_total'] = np.max(self.omega[clu])
                    df_cluster.loc[K]['min_O+_total'] = np.min(self.omega[clu])
                    df_cluster.loc[K]['mean_O+_total'] = np.mean(self.omega[clu])
                    df_cluster_idx.loc[K]['CLU+_total'] = np.sort(clu)
                if c_t == '-_total':
                    clu = clu_info['c_cluster'][c_t]
                    df_rset.loc[K]['r-_total'] = r_cl[c_t]
                    df_rset.loc[K]['sig-_total'] = sig_c[c_t][-1]
                    df_cluster.loc[K]['S-_total'] = clu_info['c_size'][c_t]
                    df_cluster.loc[K]['v-_total'] = clu_info['c_speed'][c_t]
                    df_cluster.loc[K]['max_O-_total'] = np.max(self.omega[clu])
                    df_cluster.loc[K]['min_O-_total'] = np.min(self.omega[clu])
                    df_cluster.loc[K]['mean_O-_total'] = np.mean(self.omega[clu])
                    df_cluster_idx.loc[K]['CLU-_total'] = np.sort(clu)
            df_avglast.loc[K] = clu_info['avg_dtheta_last']
            df_Thetalast.loc[K] = self.Theta_last
        TLO_info = {}
        TLO_info['Ks'] = Ks
        TLO_info['r_info'] = df_rset
        TLO_info['cluster_info'] = df_cluster 
        TLO_info['avg_dtheta'] = df_avglast
        TLO_info['Theta_last'] = df_Thetalast
        TLO_info['CLU_idx'] = df_cluster_idx

        return TLO_info
    
    def KM_space(self,K_start=0,K_end = 15,dK = 0.1):
        N = self.N
        self.dK = dK
        Ks = np.arange(K_start,K_end+dK/2,dK)
        self.Theta_last = self.Theta_ori.copy()
        df_rset = pd.DataFrame(columns=['r_mean','r0','r+','r-','r+_total','r-_total',
                                        'sig_mean','sig0','sig+','sig-','sig+_total','sig-_total'],index=Ks)
        df_cluster = pd.DataFrame(columns=['S0','S+','S-','S+_total','S-_total',
                                           'v0','v+','v-','v+_total','v-_total',
                                           'max_O0','max_O+','max_O-','max_O+_total','max_O-_total',
                                           'min_O0','min_O+','min_O-','min_O+_total','min_O-_total',
                                           'mean_O0','mean_O+','mean_O-','mean_O+_total','mean_O-_total'],index=Ks)
        # df_cluster_idx = pd.DataFrame(columns=['CLU0','CLU+','CLU-','CLU+_total','CLU-_total'],index=Ks)
        # df_avglast = pd.DataFrame(columns=range(N),index=Ks)
        # df_Thetalast = pd.DataFrame(columns=range(2*N),index=Ks)

        for K in Ks:
            self.K = K
            self.Theta_init = self.Theta_ori.copy()
            sol = self.solve_short(result_time=self.t_end*10+1-2010)
            clu_info = self.get_cluster()
            r_info = self.get_r_clu_last()
            c_type = r_info['clu_name']
            r_cl = r_info['r_clu_mean_last']
            sig_c = r_info['r_clu_std']
            df_rset.loc[K]['r_mean'] = r_info['r_total_mean'][-1]
            df_rset.loc[K]['sig_mean'] = r_info['r_total_std'][-1]
            for c_t in c_type:
                if c_t == '0':
                    clu = clu_info['c_cluster'][c_t]
                    df_rset.loc[K]['r0'] = r_cl[c_t]
                    df_rset.loc[K]['sig0'] = sig_c[c_t][-1]
                    df_cluster.loc[K]['S0'] = clu_info['c_size'][c_t]
                    df_cluster.loc[K]['v0'] = clu_info['c_speed'][c_t]
                    df_cluster.loc[K]['max_O0'] = np.max(self.omega[clu])
                    df_cluster.loc[K]['min_O0'] = np.min(self.omega[clu])
                    df_cluster.loc[K]['mean_O0'] = np.mean(self.omega[clu])
                    # df_cluster_idx.loc[K]['CLU0'] = np.sort(clu)
                if c_t == '+':
                    clu = clu_info['c_cluster'][c_t]
                    df_rset.loc[K]['r+'] = r_cl[c_t]
                    df_rset.loc[K]['sig+'] = sig_c[c_t][-1]
                    df_cluster.loc[K]['S+'] = clu_info['c_size'][c_t]
                    df_cluster.loc[K]['v+'] = clu_info['c_speed'][c_t]
                    df_cluster.loc[K]['max_O+'] = np.max(self.omega[clu])
                    df_cluster.loc[K]['min_O+'] = np.min(self.omega[clu])
                    df_cluster.loc[K]['mean_O+'] = np.mean(self.omega[clu])
                    # df_cluster_idx.loc[K]['CLU+'] = np.sort(clu)
                if c_t == '-':
                    clu = clu_info['c_cluster'][c_t]
                    df_rset.loc[K]['r-'] = r_cl[c_t]
                    df_rset.loc[K]['sig-'] = sig_c[c_t][-1]
                    df_cluster.loc[K]['S-'] = clu_info['c_size'][c_t]
                    df_cluster.loc[K]['v-'] = clu_info['c_speed'][c_t]
                    df_cluster.loc[K]['max_O-'] = np.max(self.omega[clu])
                    df_cluster.loc[K]['min_O-'] = np.min(self.omega[clu])
                    df_cluster.loc[K]['mean_O-'] = np.mean(self.omega[clu])
                    # df_cluster_idx.loc[K]['CLU-'] = np.sort(clu)
                if c_t == '+_total':
                    clu = clu_info['c_cluster'][c_t]
                    df_rset.loc[K]['r+_total'] = r_cl[c_t]
                    df_rset.loc[K]['sig+_total'] = sig_c[c_t][-1]
                    df_cluster.loc[K]['S+_total'] = clu_info['c_size'][c_t]
                    df_cluster.loc[K]['v+_total'] = clu_info['c_speed'][c_t]
                    if len(clu) != 0:
                        df_cluster.loc[K]['max_O+_total'] = np.max(self.omega[clu])
                        df_cluster.loc[K]['min_O+_total'] = np.min(self.omega[clu])
                        df_cluster.loc[K]['mean_O+_total'] = np.mean(self.omega[clu])
                    else:
                        df_cluster.loc[K]['max_O+_total'] = np.nan
                        df_cluster.loc[K]['min_O+_total'] = np.nan
                        df_cluster.loc[K]['mean_O+_total'] = np.nan
                    # df_cluster_idx.loc[K]['CLU+_total'] = np.sort(clu)
                if c_t == '-_total':
                    clu = clu_info['c_cluster'][c_t]
                    df_rset.loc[K]['r-_total'] = r_cl[c_t]
                    df_rset.loc[K]['sig-_total'] = sig_c[c_t][-1]
                    df_cluster.loc[K]['S-_total'] = clu_info['c_size'][c_t]
                    df_cluster.loc[K]['v-_total'] = clu_info['c_speed'][c_t]
                    if len(clu) != 0:
                        df_cluster.loc[K]['max_O-_total'] = np.max(self.omega[clu])
                        df_cluster.loc[K]['min_O-_total'] = np.min(self.omega[clu])
                        df_cluster.loc[K]['mean_O-_total'] = np.mean(self.omega[clu])
                    else:
                        df_cluster.loc[K]['max_O-_total'] = np.nan
                        df_cluster.loc[K]['min_O-_total'] = np.nan
                        df_cluster.loc[K]['mean_O-_total'] = np.nan

                    # df_cluster_idx.loc[K]['CLU-_total'] = np.sort(clu)
            # df_avglast.loc[K] = clu_info['avg_dtheta_last']
        KM_info = {}
        KM_info['Ks'] = Ks
        KM_info['r_info'] = df_rset
        KM_info['cluster_info'] = df_cluster 
        # KM_info['avg_dtheta'] = df_avglast
        # KM_info['Theta_last'] = df_Thetalast
        # KM_info['CLU_idx'] = df_cluster_idx
        return KM_info
    
    def MK_space(self,m_start=0,m_end = 15,dm = 0.1):
        N = self.N
        self.dm = dm
        ms = np.arange(m_start,m_end+dm/2,dm)
        self.Theta_last = self.Theta_ori.copy()
        df_rset = pd.DataFrame(columns=['r_mean','r0','r+','r-','r+_total','r-_total',
                                        'sig_mean','sig0','sig+','sig-','sig+_total','sig-_total'],index=ms)
        df_cluster = pd.DataFrame(columns=['S0','S+','S-','S+_total','S-_total',
                                           'v0','v+','v-','v+_total','v-_total',
                                           'max_O0','max_O+','max_O-','max_O+_total','max_O-_total',
                                           'min_O0','min_O+','min_O-','min_O+_total','min_O-_total',
                                           'mean_O0','mean_O+','mean_O-','mean_O+_total','mean_O-_total'],index=ms)
        # df_cluster_idx = pd.DataFrame(columns=['CLU0','CLU+','CLU-','CLU+_total','CLU-_total'],index=ms)
        # df_avglast = pd.DataFrame(columns=range(N),index=ms)
        # df_Thetalast = pd.DataFrame(columns=range(2*N),index=ms)

        for m in ms:
            self.m = m
            self.Theta_init = self.Theta_ori.copy()
            sol = self.solve()
            clu_info = self.get_cluster()
            r_info = self.get_r_clu_last()
            c_type = r_info['clu_name']
            r_cl = r_info['r_clu_mean_last']
            sig_c = r_info['r_clu_std']
            df_rset.loc[m]['r_mean'] = r_info['r_total_mean'][-1]
            df_rset.loc[m]['sig_mean'] = r_info['r_total_std'][-1]
            for c_t in c_type:
                if c_t == '0':
                    clu = clu_info['c_cluster'][c_t]
                    df_rset.loc[m]['r0'] = r_cl[c_t]
                    df_rset.loc[m]['sig0'] = sig_c[c_t][-1]
                    df_cluster.loc[m]['S0'] = clu_info['c_size'][c_t]
                    df_cluster.loc[m]['v0'] = clu_info['c_speed'][c_t]
                    df_cluster.loc[m]['max_O0'] = np.max(self.omega[clu])
                    df_cluster.loc[m]['min_O0'] = np.min(self.omega[clu])
                    df_cluster.loc[m]['mean_O0'] = np.mean(self.omega[clu])
                    # df_cluster_idx.loc[m]['CLU0'] = np.sort(clu)
                if c_t == '+':
                    clu = clu_info['c_cluster'][c_t]
                    df_rset.loc[m]['r+'] = r_cl[c_t]
                    df_rset.loc[m]['sig+'] = sig_c[c_t][-1]
                    df_cluster.loc[m]['S+'] = clu_info['c_size'][c_t]
                    df_cluster.loc[m]['v+'] = clu_info['c_speed'][c_t]
                    df_cluster.loc[m]['max_O+'] = np.max(self.omega[clu])
                    df_cluster.loc[m]['min_O+'] = np.min(self.omega[clu])
                    df_cluster.loc[m]['mean_O+'] = np.mean(self.omega[clu])
                    # df_cluster_idx.loc[m]['CLU+'] = np.sort(clu)
                if c_t == '-':
                    clu = clu_info['c_cluster'][c_t]
                    df_rset.loc[m]['r-'] = r_cl[c_t]
                    df_rset.loc[m]['sig-'] = sig_c[c_t][-1]
                    df_cluster.loc[m]['S-'] = clu_info['c_size'][c_t]
                    df_cluster.loc[m]['v-'] = clu_info['c_speed'][c_t]
                    df_cluster.loc[m]['max_O-'] = np.max(self.omega[clu])
                    df_cluster.loc[m]['min_O-'] = np.min(self.omega[clu])
                    df_cluster.loc[m]['mean_O-'] = np.mean(self.omega[clu])
                    # df_cluster_idx.loc[m]['CLU-'] = np.sort(clu)
                if c_t == '+_total':
                    clu = clu_info['c_cluster'][c_t]
                    df_rset.loc[m]['r+_total'] = r_cl[c_t]
                    df_rset.loc[m]['sig+_total'] = sig_c[c_t][-1]
                    df_cluster.loc[m]['S+_total'] = clu_info['c_size'][c_t]
                    df_cluster.loc[m]['v+_total'] = clu_info['c_speed'][c_t]
                    if len(clu) != 0:
                        df_cluster.loc[m]['max_O+_total'] = np.max(self.omega[clu])
                        df_cluster.loc[m]['min_O+_total'] = np.min(self.omega[clu])
                        df_cluster.loc[m]['mean_O+_total'] = np.mean(self.omega[clu])
                    else:
                        df_cluster.loc[m]['max_O+_total'] =np.nan
                        df_cluster.loc[m]['min_O+_total'] =np.nan
                        df_cluster.loc[m]['mean_O+_total'] =np.nan
                        # df_cluster_idx.loc[m]['CLU+_total'] = np.sort(clu)
                if c_t == '-_total':
                    clu = clu_info['c_cluster'][c_t]
                    df_rset.loc[m]['r-_total'] = r_cl[c_t]
                    df_rset.loc[m]['sig-_total'] = sig_c[c_t][-1]
                    df_cluster.loc[m]['S-_total'] = clu_info['c_size'][c_t]
                    df_cluster.loc[m]['v-_total'] = clu_info['c_speed'][c_t]
                    if len(clu) !=0:
                        df_cluster.loc[m]['max_O-_total'] = np.max(self.omega[clu])
                        df_cluster.loc[m]['min_O-_total'] = np.min(self.omega[clu])
                        df_cluster.loc[m]['mean_O-_total'] = np.mean(self.omega[clu])
                    else:
                        df_cluster.loc[m]['max_O-_total'] = np.nan
                        df_cluster.loc[m]['min_O-_total'] = np.nan
                        df_cluster.loc[m]['mean_O-_total'] = np.nan
                        # df_cluster_idx.loc[m]['CLU-_total'] = np.sort(clu)
            # df_avglast.loc[m] = clu_info['avg_dtheta_last']
            # df_Thetalast.loc[m] = self.Theta_last
        KM_info = {}
        KM_info['ms'] = ms
        KM_info['r_info'] = df_rset
        KM_info['cluster_info'] = df_cluster 
        # KM_info['avg_dtheta'] = df_avglast
        # KM_info['Theta_last'] = df_Thetalast
        # KM_info['CLU_idx'] = df_cluster_idx
        return KM_info
    
    
    def get_STEP(self,TLO_info,s_length=2):
        S0 = TLO_info['cluster_info']['S0'].dropna()
        Ks = S0.index
        dK = Ks[1] - Ks[0]

        S0_ = np.r_[0,S0]
        diff_S0 = np.diff(S0_)

        A, = np.where(diff_S0>0)
        diff_A = np.diff(A)
        diff_3, = np.where(diff_A>s_length)
        A3 =A[diff_3]
        A3_end =A[diff_3+1]
        STEP_start = Ks[A3]
        STEP_end = Ks[A3_end]-dK
        return STEP_start,STEP_end
    
    def MAKE_STEP(self,TLO_info,s_length=2,P_dK = 0.3):
        STEP_start,STEP_end = self.get_STEP(TLO_info,s_length=s_length)
        Ks = TLO_info['Ks']
        dK = Ks[1] - Ks[0]
        Ks_ = TLO_info['Theta_last'].index
        df_STEP = pd.DataFrame(columns=['S_start','S_end','Ks_step','F_RMu','F_R0u','rs_d','rs_u'],index=STEP_start)
        _,F_RMu_,_,F_R0u_ =  OSP2.Make_R_function(self.m)
        K_t = np.linspace(0,15,30000)

        for s_start,s_end in zip(STEP_start,STEP_end):
            iloc = np.searchsorted(Ks_,s_start)
            # r_0 = TLO_info['r_info']['r0'].iloc[iloc]
            O_O = TLO_info['cluster_info']['max_O0'].iloc[iloc]
            O_2O = TLO_info['cluster_info']['max_O+'].iloc[iloc]

            # r_M = TLO_info['r_info']['r_mean'].iloc[iloc]
            # r_p = TLO_info['r_info']['r+'].iloc[iloc]
            # O_O = 4/np.pi * np.sqrt(F_RMu_(s_start)*s_start/self.m)- 0.3056*1/np.sqrt(s_start*F_RMu_(s_start)*self.m**3)
            _,F_RMu,_,F_R0u =  OSP2.Make_R0_function(self.m,O_O)
            _,_,rs_d,rs_u,_,_= OSP2.get_r_sec_np(s_start,self.m,F_RMu_,samples=30)

            r0 = F_R0u(s_start)
            rp = rs_u
            # shift_O = -(s_start**2*rp*r0)/(2*self.m*(1/self.m**2+(O_O)**2))  -(s_start**2*rp*rp)/(2*self.m*(1/self.m**2+(O_O)**2))

            # O_2O = O_O-shift_O + 4/np.pi * np.sqrt(rp*s_start/self.m) - 0.3056*1/np.sqrt(s_start*rp*self.m**3)
            Ks_S = np.arange(s_start-P_dK/4,s_end+P_dK+dK/2,dK)
            # rs_dt,rs_ut,rs_d,rs_u,md,mu= OSP2.get_r_sec_np(Ks_S,self.m,F_RMu,samples=30)
            # _,_,rs_d,rs_u= OSP2.get_r_sec0_np(Ks_S,r0,shift_O,self.m,O_O,O_2O,F_RMu,samples=30)
            
            F_S,F_OR = OSP2.get_shift(O_O)
            d,u,rs_d,rs_u= OSP2.get_r_sec0_np(Ks_S,self.m,O_O,O_2O,F_R0u_,F_S,F_OR,samples=40)
            # _,_,rs_d,rs_u= OSP2.get_r_sec0_np(Ks_S,r0,shift_O,self.m,O_O,O_2O,F_RMu,samples=30)
            df_STEP.loc[s_start]['S_start'] = s_start
            df_STEP.loc[s_start]['S_end'] = s_end
            df_STEP.loc[s_start]['Ks_step'] = Ks_S
            df_STEP.loc[s_start]['F_RMu'] = F_RMu
            df_STEP.loc[s_start]['F_R0u'] = F_R0u
            df_STEP.loc[s_start]['rs_d'] = rs_d
            df_STEP.loc[s_start]['rs_u'] = rs_u
        return df_STEP

