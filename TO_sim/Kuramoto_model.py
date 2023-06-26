import numpy as np

def get_order_parameter(theta,N):
    ''' get theta and return r and theta'''
    rpsi = 1/N*np.sum(np.exp(1j*theta))
    r = np.abs(rpsi)
    psi = np.angle(rpsi)
    return r,psi

def Kuramoto_2nd_mf(Theta,t,omega,N,m,K):
    # print(f"Case m = {m}") 
    theta,dtheta,r,psi = Theta[:N],Theta[N:2*N],Theta[-2],Theta[-1]
    r,psi = get_order_parameter(theta,N)
    ddtheta = (1/m)*(-dtheta + omega + K*r*np.sin(psi - theta))
    return np.r_[dtheta,ddtheta]

def Kuramoto_1st_mf(Theta,t,omega,N,m,K):
    # print("Case m = 0")
    theta,r,psi = Theta[:N],Theta[-2],Theta[-1]
    r,psi = get_order_parameter(theta,N)
    dtheta = omega + K*r*np.sin(psi - theta)
    return np.r_[dtheta,np.zeros(N)]

def Kuramoto_2nd_mf_r(Theta,t,omega,N,m,K):
    # print(f"Case m = {m}") 
    theta,dtheta = Theta[:N],Theta[N:2*N]
    r,psi = get_order_parameter(theta,N)
    ddtheta = (1/m)*(-dtheta + omega + K*r*np.sin(psi - theta))
    return np.r_[dtheta,ddtheta],r

def Kuramoto_1st_mf_r(Theta,t,omega,N,m,K):
    # print("Case m = 0")
    theta= Theta[:N]
    r,psi = get_order_parameter(theta,N)
    dtheta = omega + K*r*np.sin(psi - theta)
    return np.r_[dtheta,np.zeros(N)],r

def Kuramoto_2nd_mf_for_solveivp(t,Theta,omega,N,m,K):
    if m != 0:
        # print(f"Case m = {m}") 
        theta,dtheta = Theta[:N],Theta[N:]
        r,psi = get_order_parameter(theta,N)
        ddtheta = (1/m)*(-dtheta + omega + K*r*np.sin(psi - theta))
        return np.r_[dtheta,ddtheta]
    else:
        # print("Case m = 0")
        theta = Theta[:N]
        r,psi = get_order_parameter(theta,N)
        dtheta = omega + K*r*np.sin(psi - theta)
        return np.r_[dtheta,np.zeros(N)]

def Kuramoto_2nd(Theta,t,omega,N,mi,m,K):
    if m != 0:
        # print(f"Case m = {m}") 
        theta,dtheta = Theta[:N],Theta[N:]
        ai,aj = np.meshgrid(theta,theta,sparse=True)
        interaction = (K/mi) * np.sin(aj-ai)
        ddtheta = (1/m)*(-dtheta + omega + interaction.sum(axis=0))
        return np.array([*dtheta,*ddtheta])
    else: 
        # print("Case m = 0")
        theta = Theta[:N]
        ai,aj = np.meshgrid(theta,theta,sparse=True)
        interaction = (K/mi) * np.sin(aj-ai)
        dtheta = omega + interaction.sum(axis=0)
        return np.array([*dtheta,*np.zeros(N)])
    
##############################################################
### Simulate Sets ####
def get_order_parameter_sets(theta,N):
    ''' get theta and return r and theta'''
    rpsi = 1/N*np.sum(np.exp(1j*theta),axis=1)
    r = np.abs(rpsi).reshape((-1,1))
    psi = np.angle(rpsi).reshape((-1,1))
    return r,psi

def Kuramoto_2nd_mf_sets(theta_sets,t,omega_sets,N,m,K_set):
    theta,dtheta = theta_sets[:,:N],theta_sets[:,N:2*N]
    r,psi = get_order_parameter_sets(theta,N)
    ddtheta = (1/m)*(-dtheta + omega_sets + K_set*r*np.sin(psi - theta))
    return np.c_[dtheta,ddtheta]

def Kuramoto_2nd_mf_sets_r(theta_sets,t,omega_sets,N,m,K_set):
    theta,dtheta = theta_sets[:,:N],theta_sets[:,N:2*N]
    r,psi = get_order_parameter_sets(theta,N)
    ddtheta = (1/m)*(-dtheta + omega_sets + K_set*r*np.sin(psi - theta))
    return np.c_[dtheta,ddtheta],r

def Kuramoto_1st_mf_sets(theta_sets,t,omega,N,m,K_set):
    # print("Case m = 0")
    theta= theta_sets[:,:N]
    r,psi = get_order_parameter_sets(theta,N)
    dtheta = omega + K_set*r*np.sin(psi - theta)
    return np.c_[dtheta,np.zeros_like(theta)]

def Kuramoto_1st_mf_sets_r(theta_sets,t,omega,N,m,K_set):
    # print("Case m = 0")
    theta= theta_sets[:,:N]
    r,psi = get_order_parameter_sets(theta,N)
    dtheta = omega + K_set*r*np.sin(psi - theta)
    return np.c_[dtheta,np.zeros_like(theta)],r