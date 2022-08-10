import numpy as np
from TO_sim.Integrator import RK4
from TO_sim.Kuramoto_model import *
from TO_sim.gen_Distribution import *

def Make_order_parameter(theta_s,N):
    rs = np.abs(np.sum(np.exp(1j*theta_s.T),axis=0))/N
    return rs

def Sol_Kuramoto_mf(N,K,m,tspan,p_theta = None, p_dtheta = None, p_omega = None,dt=0.01,mean=0, sigma =1,distribution = "Lorentzian",seed=None):
    match distribution:
        case "Lorentzian":
            theta,omega,Kc = Lorentzian(N,mean,sigma,seed)
            dtheta  =  np.zeros(N)
        case _: 
            theta,omega,Kc = Normal(N,mean,sigma,seed)
            dtheta  =  np.zeros(N)
    match p_theta,p_dtheta,p_omega:
        case  None,None,None:
            pass
        case  _:
            theta, dtheta, omega  =  p_theta, p_dtheta,p_omega
            
    t = np.arange(tspan[0],tspan[1]+dt,dt)
    result = RK4(Kuramoto_2nd_mf,np.array([*theta,*dtheta]),t,args=(omega,N,m,K))
    theta_s = result[:,:N]
    dtheta_s = result[:,N:]
    rs = Make_order_parameter(theta_s,N)
    return theta_s,dtheta_s,omega,rs,t
    
def Sol_Kuramoto(N,K,m,tspan,dt=0.01,mean=0, sigma =1,distribution = "Lorentzian",seed=None):
    match distribution:
        case "Lorentzian":
            theta,omega,Kc = Lorentzian(N,mean,sigma,seed)
        case _: 
            theta,omega,Kc = Normal(N,mean,sigma,seed)
            
    t = np.arange(tspan[0],tspan[1]+dt,dt)
    result = RK4(Kuramoto_2nd,np.array([*theta,*np.zeros(N)]),t,args=(omega,N,N-1,m,K))
    theta_s = result[:,:N]
    dtheta_s = result[:,N:]
    rs = Make_order_parameter(theta_s,N)
    return theta_s,dtheta_s,rs,t


def Sol_r_Kuramoto_mf(K,N,m,tspan,dt=0.01,mean=0, sigma =1,distribution = "Lorentzian",seed=None):
    
    match distribution:
        case "Lorentzian":
            theta,omega,Kc = Lorentzian(N,mean,sigma,seed)
        case _: 
            theta,omega,Kc = Normal(N,mean,sigma,seed)
            
    t = np.arange(tspan[0],tspan[1]+dt,dt)
    result = RK4(Kuramoto_2nd_mf,np.array([*theta,*np.zeros(N)]),t,args=(omega,N,m,K))
    theta_s = result[:,:N]
    dtheta_s = result[:,N:]
    rs = Make_order_parameter(theta_s,N)
    return rs,t

def Sol_r_Kuramoto_mf_C(K,N,m,tspan,dt=0.01,mean=0, sigma =1,distribution = "Lorentzian",seed=None):
    
    match distribution:
        case "Lorentzian":
            theta,omega,Kc = Lorentzian(N,mean,sigma,seed)
        case _: 
            theta,omega,Kc = Normal(N,mean,sigma,seed)
            
    t = np.arange(tspan[0],tspan[1]+dt,dt)
    result = RK4(Kuramoto_2nd_mf,np.array([*np.ones(N),*np.zeros(N)]),t,args=(omega,N,m,K))
    theta_s = result[:,:N]
    dtheta_s = result[:,N:]
    rs = Make_order_parameter(theta_s,N)
    return rs,t
