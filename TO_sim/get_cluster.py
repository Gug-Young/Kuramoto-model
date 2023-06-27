import numpy as np
from scipy.signal import find_peaks
from TO_sim.Sol_Kuramoto import Sol_Kuramoto_mf2 as mf2
from TO_sim.Sol_Kuramoto import Sol_Kuramoto_mf2_sets as mf2_sets


def cluster_os(avg_dtheta,N,cidx=False,p_array=False,dt=0.1):
    def to_cluster_barg(idx,peaks_new):
        C = idx,idx+1
        arg_C = peaks_new[C[0]],peaks_new[C[1]]
        cluster = np.arange(arg_C[0],arg_C[1])
        return cluster
    def to_cstability(x,diff_dtheta):
        try:
            return np.mean(diff_dtheta[x[1:]])
        except IndexError:
            return np.nan
    def to_mean_avg_d_o(x,avg_dtheta,index):
        try:
            return np.mean(avg_dtheta[index][x])
        except IndexError:
            return np.nan
    iter_time = 1500
    num = 0
    for index in range(-iter_time,0):
        arg = np.argsort(avg_dtheta[index])
        SD = avg_dtheta[index][arg]
        diff_dtheta = np.diff([SD[0],*SD,SD[-1]])
        peaks, P  = find_peaks(diff_dtheta, height=0.01)
        
        # peaks = peaks[np.where((peaks<N)&(peaks>1))]

        try:
            peaks_new = np.array([peaks[0],*peaks])
            if len(peaks) == 1:
                peaks_new = np.array([peaks[0],N])
        except IndexError:
            peaks_new = np.array([0,N])

        psize = np.diff(peaks_new)
        arg_psize = np.argsort(psize)[::-1]
        sort_psize = np.sort(psize)[::-1]
        clusters = np.array([to_cluster_barg(arg,peaks_new) for arg in arg_psize],dtype=object)[:10]
        try:
            if len(clusters) == 1:
                clusters = np.array([np.arange(peaks_new[0],N)])
            c_stability = np.array(list(map(to_cstability,clusters,[diff_dtheta]*len(clusters))))
            mean_omega = np.array(list(map(to_mean_avg_d_o,clusters,[avg_dtheta]*len(clusters),[index]*len(clusters))))
            if num==0:
                psize_array = sort_psize[:10]
                cluster_array = clusters
                c_stability_array = c_stability
                mean_omega_array = mean_omega
                arg_array = arg

                num+=1
            else:
                cluster_array = np.c_[cluster_array,clusters]
                psize_array = np.c_[psize_array,sort_psize[:10]]
                c_stability_array = np.c_[c_stability_array,c_stability]
                mean_omega_array = np.c_[mean_omega_array,mean_omega]
                arg_array = np.c_[arg_array,arg]
        except ValueError:
            pass
            

    Is_group, = np.where((np.std(psize_array,axis=1) == 0)&(psize_array[:,-1]>5))
    check = 0
    mean_group_s = np.mean(c_stability_array,axis=1)
    Is_group2, = np.where((mean_group_s<1e-4)&(psize_array[:,-1]>5))
    Is_group = np.intersect1d(Is_group,Is_group2)
    if len(Is_group)==0:
        check = 1
        mm = np.max(psize_array,axis=1) - np.min(psize_array,axis=1)
        Is_group, = np.where((mean_group_s<7e-4)&(psize_array[:,-1]>5))
        Is_group2, = np.where((mm <= 1)&(psize_array[:,-1]>5))
        Is_group = np.intersect1d(Is_group,Is_group2)

    # if len(Is_group)==0:
    #     check = 2
    #     mean_group_s = np.mean(c_stability_array,axis=1)
    #     Is_group, = np.where((mean_group_s<1e-3)&(psize_array[:,-1]>5))
    CM_O = np.mean(mean_omega_array[Is_group],axis=1)
    sCM_O = np.sort(CM_O)
    sCM_Oidx = np.argsort(CM_O)

    CM_S = np.mean(psize_array[Is_group],axis=1)[sCM_Oidx]
    if cidx == True:
        C_idx = np.array([arg[i] for i in clusters[sCM_Oidx]])
        return CM_S,sCM_O,C_idx,check,psize_array
    if p_array == True:
        return CM_S,sCM_O,C_idx,check,psize_array
    else:
        return CM_S,sCM_O,check
    
def C_rsmso(K,m,N,theta_init,omega,pdtheta,t_end=5000,dt=0.1):
    t = np.arange(0,t_end,dt)
    theta, dtheta,rs = mf2(K,m=m,N=N,t_array=t,p_theta=theta_init,p_dtheta= pdtheta,p_omega=omega,result_time = int((t_end)-(600))*int(1/dt))
    if m == 0:
        dtheta = np.c_[dtheta[0],dtheta.T].T
    r_duration = rs[-5000:]
    r = np.mean(r_duration)
    rstd = np.std(r_duration)
    rMM = np.max(r_duration)-np.min(r_duration)

    sum_time = 500*int(1/dt)
    dtheta_c = np.cumsum(dtheta,axis=0)
    avg_dtheta = (dtheta_c[sum_time:]-dtheta_c[:-sum_time])/sum_time
    dtype = [('cluster size', int), ('cluster mean phase velocity', float)]
    CM_S,CM_O = cluster_os(avg_dtheta=avg_dtheta,N=N,dt=dt)
    CSO = np.array([(S,O) for S,O in zip(CM_S,CM_O)],dtype=dtype)
    return r,rstd,rMM,CSO


def C_rsmso_set(m,K_set,N,theta_init_set,omega_set,pdtheta_set,t_end=5000,dt=0.1):
    t = np.arange(0,t_end,dt)
    _, dtheta_set,rs = mf2_sets(K_set,m=m,N=N,t_array=t,
                                p_theta=theta_init_set,p_dtheta= pdtheta_set,p_omega=omega_set,
                                result_time = int((t_end)-(200))*int(1/dt)-(150)*int(1/dt))
    if dt == 0.01:
        dtheta_set = dtheta_set[::10]

    if m == 0:
        dtheta_set = np.concatenate((dtheta_set[0].reshape(1,-1,N),dtheta_set),0)
    N_set = len(theta_init_set)
    r_duration = rs[-5000:,:]
    r = np.mean(r_duration,axis=0).reshape(-1)
    rstd = np.std(r_duration,axis=0).reshape(-1)
    rMM = (np.max(r_duration,axis=0)-np.min(r_duration,axis=0)).reshape(-1)

    sum_time = 200*int(1/dt)
    if dt == 0.01:
        sum_time = 2000
    dtheta_c = np.cumsum(dtheta_set,axis=0)
    avg_dtheta_set = (dtheta_c[sum_time:]-dtheta_c[:-sum_time])/sum_time
    dtype = [('cluster size', int), ('cluster mean phase velocity', float)]
    CSO_set = []
    check_set = [] 
    for i in range(N_set):
        CM_S,CM_O,check = cluster_os(avg_dtheta=avg_dtheta_set[:,i],N=N,dt=dt)
        CSO = np.array([(S,O) for S,O in zip(CM_S,CM_O)],dtype=dtype)
        CSO_set.append(CSO)
        check_set.append(check)
    return r,rstd,rMM,CSO_set,check_set