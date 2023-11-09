import TO_sim.Q_simul as QSIM

def MK_space(seed,N,K,m_span,t_end,start_p):
    m = 1
    m_start,m_end,dm = m_span
    Qsimul = QSIM.Q_Norm_simul(N,K=K,m = m,t_end=t_end,seed=seed,start_p=start_p)
    MK_info = Qsimul.MK_space(m_start,m_end,dm)
    ms = MK_info['ms']
    r0 = MK_info['r_info']['r0'].to_numpy(float)
    rM = MK_info['r_info']['r_mean'].to_numpy(float)
    
    rp = MK_info['r_info']['r+'].to_numpy(float)
    rp_total = MK_info['r_info']['r+_total'].to_numpy(float)
    
    rm = MK_info['r_info']['r-'].to_numpy(float)
    rm_total = MK_info['r_info']['r-_total'].to_numpy(float)
    
    sig = MK_info['r_info']['sig_mean'].to_numpy(float)
    sigPT = MK_info['r_info']['sig+_total'].to_numpy(float)
    sigMT = MK_info['r_info']['sig-_total'].to_numpy(float)

    S0 = MK_info['cluster_info']['S0'].to_numpy(float)
    SP = MK_info['cluster_info']['S+'].to_numpy(float)
    SM = MK_info['cluster_info']['S-'].to_numpy(float)
    
    V0 = MK_info['cluster_info']['v0'].to_numpy(float)
    VP = MK_info['cluster_info']['v+'].to_numpy(float)
    VM = MK_info['cluster_info']['v-'].to_numpy(float)
    
    data = {}
    data['ms']=ms
    data['r0']=r0
    data['rM']=rM
    data['r+']=rp
    data['r+_total']=rp_total
    data['r-'] =rm
    data['r-_total']=rm_total
    data['S0'] = S0
    data['S+'] = SP
    data['S-'] = SM
    
    data['V0'] = V0
    data['V+'] = VP
    data['V-'] = VM
    
    data['sig']=sig
    data['sig_+total']=sigPT
    data['sig_-total']=sigMT
    return data


def KM_space(seed,N,m,K_span,t_end,start_p):
    K = 1
    K_start,K_end,dK = K_span
    Qsimul = QSIM.Q_Norm_simul(N,K=K,m = m,t_end=t_end,seed=seed,start_p=start_p)
    KM_info = Qsimul.KM_space(K_start,K_end,dK)
    Ks = KM_info['Ks']
    r0 = KM_info['r_info']['r0'].to_numpy(float)
    rM = KM_info['r_info']['r_mean'].to_numpy(float)

    
    rp = KM_info['r_info']['r+'].to_numpy(float)
    rp_total = KM_info['r_info']['r+_total'].to_numpy(float)
    
    rm = KM_info['r_info']['r-'].to_numpy(float)
    rm_total = KM_info['r_info']['r-_total'].to_numpy(float)
    
    sig = KM_info['r_info']['sig_mean'].to_numpy(float)
    sigPT = KM_info['r_info']['sig+_total'].to_numpy(float)
    sigMT = KM_info['r_info']['sig-_total'].to_numpy(float)

    S0 = KM_info['cluster_info']['S0'].to_numpy(float)
    SP = KM_info['cluster_info']['S+'].to_numpy(float)
    SM = KM_info['cluster_info']['S-'].to_numpy(float)
    
    V0 = KM_info['cluster_info']['v0'].to_numpy(float)
    VP = KM_info['cluster_info']['v+'].to_numpy(float)
    VM = KM_info['cluster_info']['v-'].to_numpy(float)
    
    data = {}
    data['Ks']=Ks
    data['r0']=r0
    data['rM']=rM
    data['r+']=rp
    data['r+_total']=rp_total
    data['r-'] =rm
    data['r-_total']=rm_total
    data['S0'] = S0
    data['S+'] = SP
    data['S-'] = SM
    
    data['V0'] = V0
    data['V+'] = VP
    data['V-'] = VM
    
    data['sig']=sig
    data['sig_+total']=sigPT
    data['sig_-total']=sigMT
    return data


