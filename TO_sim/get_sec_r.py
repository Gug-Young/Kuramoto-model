import numpy as np
import matplotlib.pyplot as plt

from numba import njit
from numba.types import intc, CPointer, float64
from numba import cfunc, carray, jit
from scipy import LowLevelCallable
import pandas as pd

from scipy.integrate import quad
from scipy import interpolate
from scipy.integrate import quad
from TO_sim.Get_2ndR_NORM import get_r_rp,Make_R_function
from parfor import parfor
from tqdm.notebook import tqdm


def jit_integrand_function(integrand_function):
    jitted_function = jit(integrand_function, nopython=True)
    @cfunc(float64(intc, CPointer(float64)))
    def wrapped(n, xx):
        values = carray(xx, n)
        return jitted_function(values[0], values[1])
    return LowLevelCallable(wrapped.ctypes)

def jit_integrand_function2(integrand_function):
    jitted_function = jit(integrand_function, nopython=True)
    
    @cfunc(float64(intc, CPointer(float64)))
    def wrapped(n, xx):
        if n < 5:
            raise ValueError("Not enough parameters passed to function.")
        values = carray(xx, n)  # 전달된 모든 파라미터를 포함하는 배열
        return jitted_function(values[0], values[1], values[2], values[3],values[4])
    
    return LowLevelCallable(wrapped.ctypes)


def jit_integrand_function3(integrand_function):
    jitted_function = jit(integrand_function, nopython=True)
    
    @cfunc(float64(intc, CPointer(float64)))
    def wrapped(n, xx):
        if n < 5:
            raise ValueError("Not enough parameters passed to function.")
        values = carray(xx, n)  # 전달된 모든 파라미터를 포함하는 배열
        return jitted_function(values[0], values[1], values[2], values[3],values[4])
    
    return LowLevelCallable(wrapped.ctypes)



@njit
def g(x, mean, std):
    return (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(-0.5 * ((x - mean) / std) ** 2)

@njit
def g_sec(x,Or,Om):
    std = 1
    mean = -Or
    g = (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(-0.5 * ((x - mean) / std) ** 2)
    dO = abs(Or-Om)
    return np.where(x<-dO,1e-6,g)

@jit_integrand_function2
def integrand_Rl(x, X,O_O,O_p,m):
    return g(x,O_O,O_p)*np.sqrt(1-(x/X)**2)

@jit_integrand_function2
def integrand_Rd(x, X,O_O,O_p,m):
    if m == 0:
        return 0
    else:
        return X/(2*(m*x**2+1/m))*g(x,O_O,O_p)

@jit_integrand_function2
def integrand_Rl2(x, X,O_pm,shift_O,m):
    return g_sec(x,O_pm+shift_O,O_pm)*np.sqrt(1-(x/X)**2)

@jit_integrand_function2
def integrand_Rd2(x, X,O_pm,shift_O,m):
    return X/(2*(m*(x+O_pm+shift_O)**2+1/m))*g_sec(x,O_pm+shift_O,O_pm)
    # return 1/(2*(m*(x)**2+1/m))*g_sec(x,O_pm+shift_O,O_pm)


@jit_integrand_function2
def integrand_Rl3(x, X,O_pm,shift_O,m):
    return g(x,0,1)*np.sqrt(1-((x-O_pm-shift_O)/X)**2)


@jit_integrand_function2
def integrand_Rd3(x, X,O_pm,shift_O,m):
    return X/(2*(m*(x-O_pm-shift_O)**2+1/m))*g(x,0,1)


def get_r_Fun(m,Back=False,O_O=20,Inverse=True):
    Xs1 = np.logspace(-15,-3,2000)
    Xs = np.r_[Xs1,np.linspace(1e-3,60,10001)]
    R_0 = np.zeros(len(Xs))
    R_Dh = np.zeros(len(Xs))
    OPS = np.zeros(len(Xs))
    F = {}
    for i,X in enumerate(Xs):
        a = 1/np.sqrt(X*m)
        b = 4/np.pi * a - 0.3056*a**3
        b = np.where(np.where(a>1.193,1,b)>=1,1,b)
        if Back:
            omega_p = min(X,O_O)
        else:
            omega_p = b*X
        OPS[i] = omega_p
        R_0[i],err = quad(integrand_Rl, -omega_p,omega_p,args=(X,0,1,m),limit=200)
        R_Dh[i],err = quad(integrand_Rd,omega_p,np.inf,args=(X,0,1,m),limit=200)
    R_D = -2*R_Dh
    R = R_0 + R_D
    KK = 1/(R/Xs)
    K_min_idx = np.argmin(KK)
    K_d = KK[:K_min_idx]
    K_u = KK[K_min_idx:]
    R_d = R[:K_min_idx]
    R_u = R[K_min_idx:]
    RD_d = R_D[:K_min_idx]
    RD_u = R_D[K_min_idx:]
    R0_d = R_0[:K_min_idx]
    R0_u = R_0[K_min_idx:]
    OPS_d = OPS[:K_min_idx]
    OPS_u = OPS[K_min_idx:]
    F['u'] = {}
    F['u']['R'] = interpolate.interp1d(K_u,R_u,bounds_error=False)
    F['u']['R0'] = interpolate.interp1d(K_u,R0_u,bounds_error=False)
    F['u']['RD'] = interpolate.interp1d(K_u,RD_u,bounds_error=False)
    F['u']['OPS'] = interpolate.interp1d(K_u,OPS_u,bounds_error=False)
    F['d'] = {}
    F['d']['R'] = interpolate.interp1d(K_d,R_d,bounds_error=False)
    F['d']['R0'] = interpolate.interp1d(K_d,R0_d,bounds_error=False)
    F['d']['RD'] = interpolate.interp1d(K_d,RD_d,bounds_error=False)
    F['d']['OPS'] = interpolate.interp1d(K_d,OPS_d,bounds_error=False)
    F['Kc'] = KK[0]
    if Inverse:
        F['I'] = {}
        F['I']['RK'] = interpolate.interp1d(R_u,K_u,bounds_error=False)
        F['I']['ROPS'] = interpolate.interp1d(R_u,OPS_u,bounds_error=False)
    return F



def get_rp(K,r0,OP,m,MAX=False):
    rs1 = np.logspace(-6,-3,200)
    rps = np.r_[rs1,np.linspace(1e-3,(1-r0)/2,1000)] 
    RP_ls = np.nan*rps
    RP_ds = np.nan*rps

    for i,rp in enumerate(rps):
        a = 1/np.sqrt(K*rp*m)
        b = 4/np.pi * a - 0.3056*a**3
        b = np.where(np.where(a>1.193,1,b)>=1,1,b)
        if MAX:
            OPs = K*rp
        else:
            OPs = b*K*rp
        delta_P = m*K**2*r0*rp/(2*(m**2*OP**2+1)) - K**2*rp**2/(4*OP*(4*m**2*OP**2+1))
        RP_ls[i],err = quad(integrand_Rl3, OP,OP+delta_P+OPs,args=(K*rp,OP,delta_P,m),limit=200)
        RP_ds[i],err = quad(integrand_Rd, OP+delta_P+OPs,np.inf,args=(K*rp,0,1,m),limit=200)


    RP = (RP_ls-RP_ds)
    x, = np.where((RP-rps)>=0)
    try:
        rp_d = rps[x[0]]
        rp_u = rps[x[-1]]
        rp0_d = RP_ls[x[0]]
        rp0_u = RP_ls[x[-1]]
        return K,rp_d,rp_u,rp0_d,rp0_u
    except:
        return K,np.nan,np.nan,np.nan,np.nan
    
    
    

def get_rp2(K,r0,OP,m,MAX=False):
    rs1 = np.logspace(-6,-3,200)
    rps = np.r_[rs1,np.linspace(1e-3,(1-r0)/2,1000)] 
    RP_ls = np.nan*rps
    RP_ds = np.nan*rps

    for i,rp in enumerate(rps):
        a = 1/np.sqrt(K*rp*m)
        b = 4/np.pi * a - 0.3056*a**3
        b = np.where(np.where(a>1.193,1,b)>=1,1,b)
        if MAX:
            OPs = K*rp
        else:
            OPs = b*K*rp
        delta_P = m*K**2*r0*rp/(2*(m**2*OP**2+1)) + K**2*rp**2/(4*OP*(4*m**2*OP**2+1))
        RP_ls[i],err = quad(integrand_Rl3, OP,OP+delta_P+OPs,args=(K*rp,OP,delta_P,m),limit=200)
        RP_ds[i],err = quad(integrand_Rd, OP+delta_P+OPs,np.inf,args=(K*rp,0,1,m),limit=200)


    RP = (RP_ls-RP_ds)
    x, = np.where((RP-rps)>=0)
    try:
        rp_d = rps[x[0]]
        rp_u = rps[x[-1]]
        rp0_d = RP_ls[x[0]]
        rp0_u = RP_ls[x[-1]]
        return K,rp_d,rp_u,rp0_d,rp0_u
    except:
        return K,np.nan,np.nan,np.nan,np.nan
    
get_rp = np.vectorize(get_rp) 
get_rp2 = np.vectorize(get_rp2)