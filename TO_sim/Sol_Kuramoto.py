import numpy as np
from TO_sim.Integrator import RK4,RK4_sampling
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
    result = RK4(Kuramoto_2nd_mf,np.array([*theta,*dtheta]),t,args=(omega,N,m,K))
    theta_s = result[:,:N]
    dtheta_s = result[:,N:]
    rs = Make_order_parameter(theta_s,N)
    return theta_s,dtheta_s,omega,rs,t

def Sol_Kuramoto_theta_dtheta(K,N,m,t_array,p_theta = [], p_dtheta = [], p_omega = []):
    """
    멀티프로세스를 효율적으로 돌리기 위한 시스템 경량화
    """
    theta, dtheta, omega  =  p_theta, p_dtheta,p_omega
            
    result = RK4(Kuramoto_2nd_mf,np.array([*theta,*dtheta]),t_array,args=(omega,N,m,K))
    theta_s = result[:,:N]
    dtheta_s = result[:,N:]
    return theta_s,dtheta_s

def Sol_Kuramoto_sampling(N,K,m,t_array,t_sample_idx,p_theta = [], p_dtheta = [], p_omega = []):
    """
    멀티프로세스를 효율적으로 돌리기 위한 시스템 경량화
    """
    theta, dtheta, omega  =  p_theta, p_dtheta,p_omega
            
    result = RK4_sampling(Kuramoto_2nd_mf,np.array([*theta,*dtheta]),t_array,t_sample_idx,args=(omega,N,m,K))
    theta_s = result[:,:N]
    dtheta_s = result[:,N:]
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
    result = RK4(Kuramoto_2nd_mf,np.array([*theta,*dtheta]),t,args=(omega,N,m,K))
    theta_s = result[:,:N]
    rs = Make_order_parameter(theta_s,N)
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
    result = RK4(Kuramoto_2nd_mf,np.array([*theta,*dtheta]),t,args=(omega,N,m,K))
    theta_s = result[:,:N]
    rs = Make_order_parameter(theta_s,N)
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
    dtheta_s = result[:,N:]
    rs = Make_order_parameter(theta_s,N)
    return theta_s,dtheta_s,rs,t

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
    result = RK4(Kuramoto_2nd_mf,np.array([*theta,*np.zeros(N)]),t,args=(omega,N,m,K))
    theta_s = result[:,:N]
    dtheta_s = result[:,N:]
    rs = Make_order_parameter(theta_s,N)
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
    
            
    t = np.arange(tspan[0],tspan[1]+dt/2,dt)
    result = RK4(Kuramoto_2nd_mf,np.array([*np.ones(N),*np.zeros(N)]),t,args=(omega,N,m,K))
    theta_s = result[:,:N]
    dtheta_s = result[:,N:]
    rs = Make_order_parameter(theta_s,N)
    return rs,t
