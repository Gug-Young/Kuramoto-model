import numpy as np
from scipy.signal import find_peaks
from TO_sim.Sol_Kuramoto import Sol_Kuramoto_mf2 as mf2
from TO_sim.Sol_Kuramoto import Sol_Kuramoto_mf2_sets as mf2_sets
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

# def cluster_os(avg_dtheta,N,cidx=False,p_array=False,dt=0.1):
#     def to_cluster_barg(idx,peaks_new):
#         C = idx,idx+1
#         arg_C = peaks_new[C[0]],peaks_new[C[1]]
#         cluster = np.arange(arg_C[0],arg_C[1])
#         return cluster
#     def to_cstability(x,diff_dtheta):
#         try:
#             return np.mean(diff_dtheta[x[1:]])
#         except IndexError:
#             return np.nan
#     def to_mean_avg_d_o(x,avg_dtheta,index):
#         try:
#             return np.mean(avg_dtheta[index][x])
#         except IndexError:
#             return np.nan
#     iter_time = 1500
#     num = 0
#     for index in range(-iter_time,0):
#         arg = np.argsort(avg_dtheta[index])
#         SD = avg_dtheta[index][arg]
#         diff_dtheta = np.diff([SD[0],*SD,SD[-1]])
#         peaks, P  = find_peaks(diff_dtheta, height=0.01)
        
#         # peaks = peaks[np.where((peaks<N)&(peaks>1))]

#         try:
#             peaks_new = np.array([peaks[0],*peaks])
#             if len(peaks) == 1:
#                 peaks_new = np.array([peaks[0],N])
#         except IndexError:
#             peaks_new = np.array([0,N])

#         psize = np.diff(peaks_new)
#         arg_psize = np.argsort(psize)[::-1]
#         sort_psize = np.sort(psize)[::-1]
#         clusters = np.array([to_cluster_barg(arg,peaks_new) for arg in arg_psize],dtype=object)[:10]
#         try:
#             if len(clusters) == 1:
#                 clusters = np.array([np.arange(peaks_new[0],N)])
#             c_stability = np.array(list(map(to_cstability,clusters,[diff_dtheta]*len(clusters))))
#             mean_omega = np.array(list(map(to_mean_avg_d_o,clusters,[avg_dtheta]*len(clusters),[index]*len(clusters))))
#             if num==0:
#                 psize_array = sort_psize[:10]
#                 cluster_array = clusters
#                 c_stability_array = c_stability
#                 mean_omega_array = mean_omega
#                 arg_array = arg

#                 num+=1
#             else:
#                 cluster_array = np.c_[cluster_array,clusters]
#                 psize_array = np.c_[psize_array,sort_psize[:10]]
#                 c_stability_array = np.c_[c_stability_array,c_stability]
#                 mean_omega_array = np.c_[mean_omega_array,mean_omega]
#                 arg_array = np.c_[arg_array,arg]
#         except ValueError:
#             pass
            

#     Is_group, = np.where((np.std(psize_array,axis=1) == 0)&(psize_array[:,-1]>5))
#     check = 0
#     mean_group_s = np.mean(c_stability_array,axis=1)
#     Is_group2, = np.where((mean_group_s<1e-4)&(psize_array[:,-1]>5))
#     Is_group = np.intersect1d(Is_group,Is_group2)
#     if len(Is_group)==0:
#         check = 1
#         mm = np.max(psize_array,axis=1) - np.min(psize_array,axis=1)
#         Is_group, = np.where((mean_group_s<7e-4)&(psize_array[:,-1]>5))
#         Is_group2, = np.where((mm <= 1)&(psize_array[:,-1]>5))
#         Is_group = np.intersect1d(Is_group,Is_group2)

#     # if len(Is_group)==0:
#     #     check = 2
#     #     mean_group_s = np.mean(c_stability_array,axis=1)
#     #     Is_group, = np.where((mean_group_s<1e-3)&(psize_array[:,-1]>5))
#     CM_O = np.mean(mean_omega_array[Is_group],axis=1)
#     sCM_O = np.sort(CM_O)
#     sCM_Oidx = np.argsort(CM_O)

#     CM_S = np.mean(psize_array[Is_group],axis=1)[sCM_Oidx]
#     if cidx == True:
#         C_idx = np.array([arg[i] for i in clusters[sCM_Oidx]])
#         return CM_S,sCM_O,C_idx,check,psize_array
#     if p_array == True:
#         return CM_S,sCM_O,C_idx,check,psize_array
#     else:
#         return CM_S,sCM_O,check
    
def cluster_os(avg_dtheta,N,cidx=False,p_array=False,dt=0.1,peak_height = 0.03,first_sensitivity=3e-3,second_sensitivity = 4e-3):
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
        peaks, P  = find_peaks(diff_dtheta, height=peak_height)
        # peaks,  = np.where(diff_dtheta>1e-3)
        
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
    Is_group3, = np.where((np.std(psize_array,axis=1) == 0))
    check = 0
    mean_group_s = np.mean(c_stability_array,axis=1)
    Is_group2, = np.where((mean_group_s<first_sensitivity)&(psize_array[:,-1]>5))
    Is_group = np.intersect1d(Is_group,Is_group2)
    if len(Is_group)==0:
        check = 1
        mm = np.max(psize_array,axis=1) - np.min(psize_array,axis=1)
        Is_group, = np.where((mean_group_s<second_sensitivity)&(psize_array[:,-1]>5))
        Is_group2, = np.where((mm <= 2)&(psize_array[:,-1]>5))
        Is_group = np.intersect1d(Is_group,Is_group2)

    # if len(Is_group)==0:
    #     check = 2
    #     mean_group_s = np.mean(c_stability_array,axis=1)
    #     Is_group, = np.where((mean_group_s<1e-3)&(psize_array[:,-1]>5))
    CM_O = np.mean(mean_omega_array[Is_group],axis=1)
    sCM_O = np.sort(CM_O)
    sCM_Oidx = np.argsort(CM_O)

    # CM_S = np.mean(psize_array[Is_group],axis=1)[sCM_Oidx]
    CM_S = np.mean(psize_array[Is_group],axis=1)
    C_idx = np.array([arg[i] for i in clusters[Is_group]])

    if cidx == True:
        C_idx = np.array([arg[i] for i in clusters[Is_group]])
        return CM_S,CM_O,C_idx,check,psize_array,cluster_array,Is_group3
    if p_array == True:
        return CM_S,CM_O,C_idx,check,psize_array
    else:
        return CM_S,CM_O,C_idx,check


    
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

    if m == 0:
        dtheta_set = np.concatenate((dtheta_set[0].reshape(1,-1,N),dtheta_set),0)
    N_set = len(theta_init_set)
    r_duration = rs[-5000:,:]
    r = np.mean(r_duration,axis=0).reshape(-1)
    rstd = np.std(r_duration,axis=0).reshape(-1)
    rMM = (np.max(r_duration,axis=0)-np.min(r_duration,axis=0)).reshape(-1)

    sum_time = 200*int(1/dt)
    if dt <= 0.01:
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

def peaks_(x,h=1e-3,N=500):
    A, = np.where(x>h)
    A =  np.r_[0,A,N]
    diff_A = np.diff(A)
    sort_ = np.sort(diff_A)[::-1][:10]
    argsort_ = np.argsort(diff_A)[::-1]
    C_start = A[argsort_][:10]
    C_end = A[argsort_+1][:10]
    return sort_,C_start,C_end
# AVG = avg_dtheta_set[-1500:,K_want]

def cluster_os_new2(AVG,height=1e-2,c_std = 3,check=2e-4,c_size=3,N=500,omega=[]):
    num = 0
    C_size = []
    C_start = []
    C_end = []
    sort_avg = np.sort(AVG,axis=1)
    sort_argavg = np.argsort(AVG,axis=1)
    diff_avg = np.diff(np.c_[sort_avg[:,0],sort_avg],axis=1)
    min_len = 10
    for A,c_start,c_end in [peaks_(x,height,N=N) for x in diff_avg]:
        min_len = min(min_len,len(A))
        C_size.append(A)
        C_start.append(c_start)
        C_end.append(c_end)
    C_size = np.array([c_[:min_len] for c_ in C_size]).T
    C_start = np.array([c_[:min_len] for c_ in C_start]).T
    C_end = np.array([c_[:min_len] for c_ in C_end]).T
    Is_group, = np.where((np.std(C_start,axis=1)<c_std)&(A[:min_len]>c_size))
    arg = sort_argavg[-1]

    cluster = np.array([np.arange(c_i,c_j,1) for c_i,c_j in zip(C_start[Is_group,-1],C_end[Is_group,-1])])
    check_ = [np.mean(diff_avg[-1,c_[1:]])<check for c_ in cluster]
    cluster = cluster[check_]
    C_s,C_e = [],[]
    CMO = []
    for c_ in cluster:
        c_s = np.min(arg[c_])
        c_e = np.max(arg[c_])
        C_s.append(c_s)
        C_e.append(c_e)
        mean_omega = np.mean(omega[c_s:c_e])
        CMO.append(mean_omega)
    Is_group = Is_group[check_]
    omega_s = [omega[c_s] for c_s in C_s]
    omega_e = [omega[c_e-1] for c_e in C_e]
    CMP = np.array([np.mean(sort_avg[0,c_]) for c_ in cluster])

    CS = np.array([len(c_) for c_ in cluster])
    cluster = np.array([sort_argavg[-1,c_] for c_ in cluster])
    return CS,CMP,cluster,omega_s,omega_e,CMO,Is_group,C_s,C_e

def C_rsmso_set_new(m,K_set,N,theta_init_set,omega_set,pdtheta_set,t_end=5000,dt=0.1):
    t = np.arange(0,t_end,dt)
    omega = omega_set[0]
    theta_set, dtheta_set,rs = mf2_sets(K_set,N=N,m=m,t_array=t,
                                    p_theta=theta_init_set,p_dtheta= pdtheta_set,p_omega=omega_set,
                                    result_time =  int((t_end)-(350))*int(1/dt))
    if m == 0:
        dtheta_set = np.concatenate((dtheta_set[0].reshape(1,-1,N),dtheta_set),0)
    sum_time = 1500
    dtheta_c = np.cumsum(dtheta_set,axis=0)
    avg_dtheta_set = (dtheta_c[sum_time:]-dtheta_c[:-sum_time])/sum_time
    Ks = K_set.reshape(-1)
    r_duration = rs.T[0].T[-sum_time:]
    mean_rs = np.mean(r_duration,axis=0)
    mean_rs_std = np.std(r_duration,axis=0)
    rMM = (np.max(r_duration,axis=0)-np.min(r_duration,axis=0)).reshape(-1)
    N_set = len(theta_init_set)
    dtype = [('cluster size', int), ('cluster mean phase velocity', float)]
    dtype2 = [('cluster size', int), ('cluster mean natural frequency', float)]
    CSMP_set = []
    CSMO_set = []
    check_set = [] 
    cluster_set = []
    C_s_set = []
    C_e_set = []
    c_threshold = np.where(mean_rs<0.1,1e-5,3e-4)
    error_set = []
    C_omega_s = []
    C_omega_e = []


    Ks =K_set.reshape(-1)
    for i in range(N_set):
        AVG = avg_dtheta_set[-1500:,i]
        c_check = c_threshold[i]
        try:
            CS,CMP,cluster,omega_s,omega_e,CMO,Is_group,C_s,C_e = cluster_os_new2(AVG=AVG,check=c_check,c_size=5,N=N,omega=omega)
            CSMP = np.array([(S,P) for S,P in zip(CS,CMP)],dtype=dtype)
            CSMO = np.array([(S,O) for S,O in zip(CS,CMO)],dtype=dtype2)
            CSMP_set.append(CSMP)
            CSMO_set.append(CSMO)
            error_set.append(np.nan)
            C_omega_s.append(omega_s)
            C_omega_e.append(omega_e)
            C_s_set.append(C_s)
            C_e_set.append(C_e)
        except ValueError:
            CS = []
            CMP = []
            CMO = []
            CSMP = np.array([(S,P) for S,P in zip(CS,CMP)],dtype=dtype)
            CSMO = np.array([(S,O) for S,O in zip(CS,CMO)],dtype=dtype)

            CSMP_set.append(CSMP)
            CSMO_set.append(CSMO)
            error_set.append(Ks[i])
            C_omega_s.append(np.nan)
            C_omega_e.append(np.nan)
            C_s_set.append([])
            C_e_set.append([])
    return mean_rs,mean_rs_std,rMM,CSMP_set,CSMO_set,error_set,C_s_set,C_e_set