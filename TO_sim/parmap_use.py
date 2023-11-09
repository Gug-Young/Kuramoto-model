import Q_simul as QSIM

def MK_space(seed,N,m_span,t_end,start_p):
    m = 1
    K = 1
    m_start,m_end,dm = m_span
    Qsimul = QSIM.Q_Norm_simul(N,K=K,m = m,t_end=t_end,seed=seed,start_p=start_p)
    KM_info = Qsimul.MK_space(m_start,m_end,dm)
    
