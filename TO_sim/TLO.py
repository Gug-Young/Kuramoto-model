from TO_sim.Sol_Kuramoto import Sol_Kuramoto_mf2 as mf2
from TO_sim.Sol_Kuramoto import Sol_Kuramoto_mf2_sets_not0 as mf2_sets_TLO
import numpy as np
import pandas as pd
from TO_sim.get_cluster import cluster_os_new2


def hysterisis(df_Km,sets,theta_col,dtheta_col,K,m,N,omega):
    theta,dtheta,rs = sets
    r_duration = rs[-5000:]
    r = np.mean(r_duration,axis=0)
    rstd = np.std(r_duration,axis=0)
    rMM = (np.max(r_duration,axis=0)-np.min(r_duration,axis=0))

    sum_time = 1500
    dtheta_c = np.cumsum(dtheta,axis=0)
    avg_dtheta = (dtheta_c[sum_time:]-dtheta_c[:-sum_time])/sum_time

    c_threshold = np.where(r<0.1,1e-5,1e-4)
    CS,CMP,cluster,omega_s,omega_e,CMO,Is_group,C_s,C_e = cluster_os_new2(AVG=avg_dtheta,check=c_threshold,c_size=5,N=N,omega=omega)
    dtype = [('cluster size', int), ('cluster mean phase velocity', float)]
    dtype2 = [('cluster size', int), ('cluster mean natural frequency', float)]
    CSMP = np.array([(S,P) for S,P in zip(CS,CMP)],dtype=dtype)
    CSMO = np.array([(S,O) for S,O in zip(CS,CMO)],dtype=dtype2)
    cluster = np.array(cluster)
    last_theta = theta[-1]
    last_dtheta = dtheta[-1]
    df_Km.loc[(K,m),"r"] = r
    df_Km.loc[(K,m),"rstd"] = rstd
    df_Km.loc[(K,m),"rMM"] = rMM
    df_Km.loc[(K,m),'error'] = np.nan
    for i in range(len(CSMO)):
        df_Km.loc[(K,m),f'c{i}'] = CSMP['cluster size'][i]
        df_Km.loc[(K,m),f'c{i} phase vel'] = CSMP['cluster mean phase velocity'][i]
        df_Km.loc[(K,m),f'c{i} omega'] = CSMO['cluster mean natural frequency'][i]
        df_Km.loc[(K,m),f'c{i} list'] = ' '.join(map(str,cluster[i]))
    df_Km.loc[(K,m),theta_col] = last_theta
    df_Km.loc[(K,m),dtheta_col] = last_dtheta
    return (last_theta,last_dtheta)

def make_new_df(K_start,m_start,N = 500):
    cols ={'r':[0],'rstd':[0],'rMM':[0],'error':[0]}
    theta_col = []
    dtheta_col = []
    for i in range(10):
        cols[f'c{i}'] = np.nan
    for i in range(10):
        cols[f'c{i} phase vel'] = np.nan
    for i in range(10):
        cols[f'c{i} omega'] = np.nan
    for i in range(10):
        cols[f'c{i} list'] = np.nan

    for i in range(N):
        s = 'theta'+f'{i}'.zfill(3)
        theta_col.append(s)
        cols[s]= np.nan

    for i in range(N):
        s = 'dtheta'+f'{i}'.zfill(3)
        dtheta_col.append(s)
        cols[s]= np.nan

    cols['K'] = K_start
    cols['m'] = m_start
    df = pd.DataFrame(columns=cols.keys())
    df_Km = df.set_index(['K','m'])
    for i in range(10):
        df_Km[f'c{i} list'] =df_Km[f'c{i} list'].astype(object)
    return df_Km,theta_col,dtheta_col

def TLO(m,theta_init,dtheta_init,omega,Ks,N,t_end=500,dt = 0.1):
    df_Km,theta_col,dtheta_col = make_new_df(0,m,N = N)
    t = np.arange(0,t_end,dt)
    K = Ks[0]
    theta, dtheta,rs = mf2(K,N=N,m=m,t_array=t,p_theta=theta_init,p_dtheta= dtheta_init,p_omega=omega,result_time = int((t_end)-(350))*int(1/dt))
    if m == 0:
        dtheta = np.c_[dtheta[0],dtheta.T].T
    last_theta,last_dtheta = hysterisis(df_Km,(theta, dtheta,rs),theta_col,dtheta_col,K,m,N,omega)
    for K in Ks[1:]:
        theta, dtheta,rs = mf2(K,N=N,m=m,t_array=t,p_theta=last_theta,p_dtheta= last_dtheta,p_omega=omega,result_time = int((t_end)-(350))*int(1/dt))
        if m == 0:
            dtheta = np.c_[dtheta[0],dtheta.T].T
        last_theta,last_dtheta = hysterisis(df_Km,(theta, dtheta,rs),theta_col,dtheta_col,K,m,N,omega)
    return df_Km


def hysterisis_col(df_Km,sets,theta_col,dtheta_col,K,m_set,N,omega):
    theta_set,dtheta_set,rs = sets
    r_duration = rs[-5000:,:]
    r = np.mean(r_duration,axis=0)
    rstd = np.std(r_duration,axis=0)
    rMM = (np.max(r_duration,axis=0)-np.min(r_duration,axis=0))

    sum_time = 1500
    dtheta_c = np.cumsum(dtheta_set,axis=0)
    avg_dtheta_set = (dtheta_c[sum_time:]-dtheta_c[:-sum_time])/sum_time

    r_duration = rs.T[0].T[-sum_time:]
    mean_rs = np.mean(r_duration,axis=0)

    dtype = [('cluster size', int), ('cluster mean phase velocity', float)]
    dtype2 = [('cluster size', int), ('cluster mean natural frequency', float)]

    c_threshold = np.where(mean_rs<0.05,1e-5,3e-4)
    last_theta = theta_set[-1]
    last_dtheta = dtheta_set[-1]
    for i,m in enumerate(m_set.reshape(-1)):
            AVG = avg_dtheta_set[-1500:,i]
            df_Km.loc[(K,m),"r"] = r[i]
            df_Km.loc[(K,m),"rstd"] = rstd[i]
            df_Km.loc[(K,m),"rMM"] = rMM[i]
            c_check = c_threshold[i]
            CS,CMP,cluster,omega_s,omega_e,CMO,Is_group,C_s,C_e = cluster_os_new2(AVG=AVG,check=c_check,c_size=5,N=N,omega=omega)
            CSMP = np.array([(S,P) for S,P in zip(CS,CMP)],dtype=dtype)
            CSMO = np.array([(S,O) for S,O in zip(CS,CMO)],dtype=dtype2)
            for i in range(len(CSMP)):
                    df_Km.loc[(K,m),f'c{i}'] = CSMP['cluster size'][i]
                    df_Km.loc[(K,m),f'c{i} phase vel'] = CSMP['cluster mean phase velocity'][i]
                    df_Km.loc[(K,m),f'c{i} omega'] = CSMO['cluster mean natural frequency'][i]
                    df_Km.loc[(K,m),f'c{i} list'] = ' '.join(map(str,cluster[i]))
            df_Km.loc[(K,m),theta_col] = last_theta[i]
            df_Km.loc[(K,m),dtheta_col] = last_dtheta[i]
    return (last_theta,last_dtheta)

def TLO_col(m_set,theta_init_set,omega_set,Ks,N,t_end=500,dt = 0.1):
    df_Km,theta_col,dtheta_col = make_new_df(0,m_set.reshape(-1)[0],N = N)
    omega = omega_set[0]
    t = np.arange(0,t_end,dt)
    K = Ks[0]
    theta_set, dtheta_set,rs = mf2_sets_TLO(m_set = m_set,N=N,K=K,t_array=t,p_theta=theta_init_set,p_dtheta= 0*omega_set,p_omega=omega_set,result_time=int((t_end-350)*(1/dt)))
    last_theta,last_dtheta = hysterisis_col(df_Km,(theta_set,dtheta_set,rs),theta_col,dtheta_col,K,m_set,N,omega)
    for K in Ks[1:]:
        theta_set,dtheta_set,rs = mf2_sets_TLO(m_set = m_set,N=N,K=K,t_array=t,p_theta=last_theta,p_dtheta= last_dtheta,p_omega=omega_set,result_time = int((t_end)-(350))*int(1/dt))
        last_theta,last_dtheta = hysterisis_col(df_Km,(theta_set,dtheta_set,rs),theta_col,dtheta_col,K,m_set,N,omega)
    return df_Km