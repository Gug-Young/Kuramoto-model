import numpy as np

def get_order_parameter(theta,N):
    ''' get theta and return r and theta'''
    rpsi = 1/N*np.sum(np.exp(1j*theta))
    r = np.abs(rpsi)
    psi = np.angle(rpsi)
    return r,psi

def Kuramoto_2nd_mf(Theta,t,omega,N,m,K):
    if m != 0:
        # print(f"Case m = {m}") 
        theta,dtheta = Theta[:N],Theta[N:]
        r,psi = get_order_parameter(theta,N)
        ddtheta = (1/m)*(-dtheta + omega + K*r*np.sin(psi - theta))
        return np.array([*dtheta,*ddtheta])
    else:
        # print("Case m = 0")
        theta = Theta[:N]
        r,psi = get_order_parameter(theta,N)
        dtheta = omega + K*r*np.sin(psi - theta)
        return np.array([*dtheta,*np.zeros(N)])
        

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