import numpy as np
from TO_sim.Integrator import RK4,RK4_sampling,RK4_r, RK4_r_sets
from TO_sim.Kuramoto_model import *
from TO_sim.gen_Distribution import *

def Make_order_parameter(theta_s,N):
    rs = np.abs(np.sum(np.exp(1j*theta_s.T),axis=0))/N
    return rs
def Sol_Kuramoto_mf(N,K,m,tspan,p_theta = [], p_dtheta = [], p_omega = [],dt=0.01,mean=0, sigma =1,distribution = "Lorentzian",seed=None):
    if tuple(map(len,(p_theta,p_dtheta,p_omega)))==(0,0,0):
        if distribution == "Lorentzian":
            theta,omega,Kc = Lorentzian(N,mean,sigma,seed)
            dtheta  =  np.zeros(N)
        elif distribution == "Quantile Lorentzian":
            theta,omega,Kc = Quantile_Lorentzian(N,mean,sigma,seed)
            dtheta  =  np.zeros(N)
        elif distribution == "Quantile Normal":
            theta,omega,Kc = Quantile_Normal(N,mean,sigma,seed)
            dtheta  =  np.zeros(N)
        else:
            theta,omega,Kc = Normal(N,mean,sigma,seed)
            dtheta  =  np.zeros(N)
    else:
        theta, dtheta, omega  =  p_theta, p_dtheta,p_omega
    
    t = np.arange(tspan[0],tspan[1]+dt/2,dt)
    if m==0:
        function = Kuramoto_1st_mf_r
    else:
        function = Kuramoto_2nd_mf_r
    result,rs = RK4_r(function,np.r_[theta,dtheta],t,args=(omega,N,m,K))
    theta_s = result[:,:N]
    dtheta_s = result[:,N:2*N]
    return theta_s,dtheta_s,omega,rs,t

def Sol_Kuramoto_mf2(K,N,m,t_array,p_theta = [], p_dtheta = [], p_omega = [],result_time = 0):
    theta, dtheta, omega  =  p_theta, p_dtheta,p_omega
    if m==0:
        function = Kuramoto_1st_mf_r
    else:
        function = Kuramoto_2nd_mf_r
    result,rs = RK4_r(function,np.r_[theta,dtheta],t_array,args=(omega,N,m,K),result_time=result_time)
    theta_s = result[:,:N]
    dtheta_s = result[:,N:2*N]
    if m == 0:
        dt = t_array[1]-t_array[0]
        dtheta_s = np.diff(theta_s/dt,axis=0)
    return theta_s,dtheta_s,rs


def Sol_Kuramoto_mK(mK,N,t_array,p_theta = [], p_dtheta = [], p_omega = []):
    m,K = mK
    theta, dtheta, omega  =  p_theta, p_dtheta,p_omega
    if m==0:
        function = Kuramoto_1st_mf_r
    else:
        function = Kuramoto_2nd_mf_r
    result,rs = RK4_r(function,np.r_[theta,dtheta],t_array,args=(omega,N,m,K))
    theta_s = result[:,:N]
    dtheta_s = result[:,N:2*N]
    if m == 0:
        dt = t_array[1]-t_array[0]
        dtheta_s = np.diff(theta_s/dt,axis=0)
    return theta_s,dtheta_s,rs

def Sol_Kuramoto_r(K,N,m,t_array,p_theta = [], p_dtheta = [], p_omega = []):
    """
    멀티프로세스를 효율적으로 돌리기 위한 시스템 경량화
    """
    theta, dtheta, omega  =  p_theta, p_dtheta,p_omega
    n = len(t_array)
    rs = np.zeros((n))
    r,psi = get_order_parameter(theta,N)
    rs[0] = r
    h = t_array[1] - t_array[0]
    args=(omega,N,m,K)
    y_temp = np.r_[theta,dtheta]
    if m==0:
        f = Kuramoto_1st_mf_r
    else:
        f = Kuramoto_2nd_mf_r
    for i in range(n - 1):
        t_temp = t_array[i]
        k1,r = f(y_temp, t_temp, *args)
        k2,_ = f(y_temp + k1 * h * 0.5, t_temp + h * 0.5, *args)
        k3,_ = f(y_temp + k2 * h * 0.5, t_temp + h * 0.5, *args)
        k4,_ = f(y_temp + k3 * h, t_temp + h, *args)
        y_temp= y_temp + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        rs[i+1] = r

    return rs


def Sol_Kuramoto_r_set(data,N,m,t_array):
    """
    멀티프로세스를 효율적으로 돌리기 위한 시스템 경량화
    data = K, p_theta,p_dtheta,p_omega
    """
    K,theta, dtheta, omega  =  data
    n = len(t_array)
    rs = np.zeros((n))
    r,psi = get_order_parameter(theta,N)
    rs[0] = r
    h = t_array[1] - t_array[0]
    args=(omega,N,m,K)
    y_temp = np.r_[theta,dtheta]
    if m==0:
        f = Kuramoto_1st_mf_r
    else:
        f = Kuramoto_2nd_mf_r
    for i in range(n - 1):
        t_temp = t_array[i]
        k1,r = f(y_temp, t_temp, *args)
        k2,_ = f(y_temp + k1 * h * 0.5, t_temp + h * 0.5, *args)
        k3,_ = f(y_temp + k2 * h * 0.5, t_temp + h * 0.5, *args)
        k4,_ = f(y_temp + k3 * h, t_temp + h, *args)
        y_temp= y_temp + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        rs[i+1] = r

    return rs

def Sol_Kuramoto_rV(K,N,m,t_array,p_theta = [], p_dtheta = [], p_omega = []):
    """
    멀티프로세스를 효율적으로 돌리기 위한 시스템 경량화
    """
    theta, dtheta, omega  =  p_theta, p_dtheta,p_omega
    n = len(t_array)
    rs = np.zeros((n))
    Vs = np.zeros((n))
    r,psi = get_order_parameter(theta,N)
    rs[0] = r
    Vs[0] = np.var(dtheta)
    h = t_array[1] - t_array[0]
    args=(omega,N,m,K)
    y_temp = np.r_[theta,dtheta]
    if m==0:
        f = Kuramoto_1st_mf_r
    else:
        f = Kuramoto_2nd_mf_r
    for i in range(n - 1):
        t_temp = t_array[i]
        k1,r = f(y_temp, t_temp, *args)
        k2,_ = f(y_temp + k1 * h * 0.5, t_temp + h * 0.5, *args)
        k3,_ = f(y_temp + k2 * h * 0.5, t_temp + h * 0.5, *args)
        k4,_ = f(y_temp + k3 * h, t_temp + h, *args)
        y_temp= y_temp + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        rs[i+1] = r
        Vs[i+1] = np.var(y_temp[N:2*N])
    return rs,Vs



def Sol_Kuramoto_theta_dtheta(K,N,m,t_array,p_theta = [], p_dtheta = [], p_omega = []):
    """
    멀티프로세스를 효율적으로 돌리기 위한 시스템 경량화
    """
    theta, dtheta, omega  =  p_theta, p_dtheta,p_omega
    if m==0:
        function = Kuramoto_1st_mf
    else:
        function = Kuramoto_2nd_mf
    r,psi = get_order_parameter(theta,N)
    result = RK4(function,np.r_[theta,dtheta],t_array,args=(omega,N,m,K))        
    theta_s = result[:,:N]
    dtheta_s = result[:,N:2*N]
    return theta_s,dtheta_s



def Sol_Kuramoto_sampling(N,K,m,t_array,t_sample_idx,p_theta = [], p_dtheta = [], p_omega = []):
    """
    멀티프로세스를 효율적으로 돌리기 위한 시스템 경량화
    """
    theta, dtheta, omega  =  p_theta, p_dtheta,p_omega
    if m==0:
        function = Kuramoto_1st_mf
    else:
        function = Kuramoto_2nd_mf
    result = RK4_sampling(function,np.r_[theta,dtheta],t_array,t_sample_idx,args=(omega,N,m,K))    
    theta_s = result[:,:N]
    dtheta_s = result[:,N:2*N]
    return theta_s,dtheta_s


def Sol_Kuramoto_mf_r_t(N,K,m,tspan,p_theta = [], p_dtheta = [], p_omega = [],dt=0.01,mean=0, sigma =1,distribution = "Lorentzian",seed=None):
    if tuple(map(len,(p_theta,p_dtheta,p_omega)))==(0,0,0):
        if distribution == "Lorentzian":
            theta,omega,Kc = Lorentzian(N,mean,sigma,seed)
            dtheta  =  np.zeros(N)
        elif distribution == "Quantile Lorentzian":
            theta,omega,Kc = Quantile_Lorentzian(N,mean,sigma,seed)
            dtheta  =  np.zeros(N)
        elif distribution == "Quantile Normal":
            theta,omega,Kc = Quantile_Normal(N,mean,sigma,seed)
            dtheta  =  np.zeros(N)
        else:
            theta,omega,Kc = Normal(N,mean,sigma,seed)
            dtheta  =  np.zeros(N)
    else:
        theta, dtheta, omega  =  p_theta, p_dtheta,p_omega
            
    t = np.arange(tspan[0],tspan[1]+dt/2,dt)

    if m==0:
        function = Kuramoto_1st_mf
    else:
        function = Kuramoto_2nd_mf
    result,rs = RK4_r(function,np.r_[theta,dtheta],t,args=(omega,N,m,K))
    theta_s = result[:,:N]
    dtheta_s = result[:,N:2*N] 
    return rs,t

def Sol_Kuramoto_mf_rb_t(N,K,m,tspan,p_theta = [], p_dtheta = [], p_omega = [],dt=0.01,mean=0, sigma =1,distribution = "Lorentzian",seed=None):
    if tuple(map(len,(p_theta,p_dtheta,p_omega)))==(0,0,0):
        if distribution == "Lorentzian":
            theta,omega,Kc = Lorentzian(N,mean,sigma,seed)
            dtheta  =  np.zeros(N)
        elif distribution == "Quantile Lorentzian":
            theta,omega,Kc = Quantile_Lorentzian(N,mean,sigma,seed)
            dtheta  =  np.zeros(N)
        elif distribution == "Quantile Normal":
            theta,omega,Kc = Quantile_Normal(N,mean,sigma,seed)
            dtheta  =  np.zeros(N)
        else:
            theta,omega,Kc = Normal(N,mean,sigma,seed)
            dtheta  =  np.zeros(N)
    else:
        theta, dtheta, omega  =  p_theta, p_dtheta,p_omega
    theta = np.ones(N)        
    t = np.arange(tspan[0],tspan[1]+dt/2,dt)
    result,rs = RK4_r(function,np.r_[theta,dtheta],t,args=(omega,N,m,K))    
    return rs,t


def Sol_r_Kuramoto_mf(K,N,m,tspan,dt=0.01,mean=0, sigma =1,distribution = "Lorentzian",seed=None):
    if distribution == "Lorentzian":
        theta,omega,Kc = Lorentzian(N,mean,sigma,seed)
        dtheta  =  np.zeros(N)
    elif distribution == "Quantile Lorentzian":
        theta,omega,Kc = Quantile_Lorentzian(N,mean,sigma,seed)
        dtheta  =  np.zeros(N)
    elif distribution == "Quantile Normal":
        theta,omega,Kc = Quantile_Normal(N,mean,sigma,seed)
        dtheta  =  np.zeros(N)
    else:
        theta,omega,Kc = Normal(N,mean,sigma,seed)
        dtheta  =  np.zeros(N)
    t = np.arange(tspan[0],tspan[1]+dt/2,dt)
    
    if m==0:
        function = Kuramoto_1st_mf
    else:
        function = Kuramoto_2nd_mf
    result,rs = RK4_r(function,np.r_[theta,dtheta],t,args=(omega,N,m,K))   
    return rs,t

def Sol_r_Kuramoto_mf_C(K,N,m,tspan,dt=0.01,mean=0, sigma =1,distribution = "Lorentzian",seed=None):
    if distribution == "Lorentzian":
        theta,omega,Kc = Lorentzian(N,mean,sigma,seed)
        dtheta  =  np.zeros(N)
    elif distribution == "Quantile Lorentzian":
        theta,omega,Kc = Quantile_Lorentzian(N,mean,sigma,seed)
        dtheta  =  np.zeros(N)
    elif distribution == "Quantile Normal":
        theta,omega,Kc = Quantile_Normal(N,mean,sigma,seed)
        dtheta  =  np.zeros(N)
    else:
        theta,omega,Kc = Normal(N,mean,sigma,seed)
        dtheta  =  np.zeros(N)
    if m==0:
        function = Kuramoto_1st_mf
    else:
        function = Kuramoto_2nd_mf
    t = np.arange(tspan[0],tspan[1]+dt/2,dt)
    r,psi = get_order_parameter(theta,N)
    result = RK4(function,np.r_[np.ones(N),np.zeros(N),r,psi],t,args=(omega,N,m,K))
    rs = result[:,-2]
    return rs,t


    
def Sol_Kuramoto(N,K,m,tspan,dt=0.01,mean=0, sigma =1,distribution = "Lorentzian",seed=None):
    if distribution == "Lorentzian":
        theta,omega,Kc = Lorentzian(N,mean,sigma,seed)
        dtheta  =  np.zeros(N)
    elif distribution == "Quantile Lorentzian":
        theta,omega,Kc = Quantile_Lorentzian(N,mean,sigma,seed)
        dtheta  =  np.zeros(N)
    elif distribution == "Quantile Normal":
        theta,omega,Kc = Quantile_Normal(N,mean,sigma,seed)
        dtheta  =  np.zeros(N)
    else:
        theta,omega,Kc = Normal(N,mean,sigma,seed)
        dtheta  =  np.zeros(N)
    
    t = np.arange(tspan[0],tspan[1]+dt,dt)

    result = RK4(Kuramoto_2nd,np.array([*theta,*np.zeros(N)]),t,args=(omega,N,N-1,m,K))
    theta_s = result[:,:N]
    dtheta_s = result[:,N:2*N]
    rs = result[:,-2]
    return theta_s,dtheta_s,rs,t


def Sol_Kuramoto_mf2_sets(K_set,N,m,t_array,p_theta = [], p_dtheta = [], p_omega = [],result_time = 0):
    theta, dtheta, omega  =  p_theta, p_dtheta,p_omega
    if m==0:
        function = Kuramoto_1st_mf_sets_r
    else:
        function = Kuramoto_2nd_mf_sets_r
    result,rs = RK4_r_sets(function,np.c_[theta,dtheta],t_array,args=(omega,N,m,K_set),result_time=result_time)
    theta_s = result[:,:,:N]
    dtheta_s = result[:,:,N:2*N]
    if m == 0:
        dt = t_array[1]-t_array[0]
        dtheta_s = np.diff(theta_s/dt,axis=0)
    return theta_s,dtheta_s,rs