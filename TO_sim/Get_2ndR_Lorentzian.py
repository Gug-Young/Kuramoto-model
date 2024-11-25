import numpy as np
import scipy.integrate as si
from numba import cfunc, carray, jit
import numba
from numba.types import intc, CPointer, float64
from scipy import LowLevelCallable
from scipy.stats import norm
from numba import njit, prange
from scipy.integrate import quad
from scipy import interpolate
from numba import njit
import numpy as np

@njit
def g(x, mean, std):
    return std/(np.pi*((x-mean)**2+std**2))


@njit
def g_sec(x,Or,Om):
    std = 1
    mean = -Or
    g = std/(np.pi*((x-mean)**2+std**2))
    dO = abs(Or-Om)
    return np.where(x<-dO,1e-6,g)

def jit_integrand_function(integrand_function):
    jitted_function = numba.jit(integrand_function, nopython=True)
    @cfunc(float64(intc, CPointer(float64)))
    def wrapped(n, xx):
        values = carray(xx, n)
        return jitted_function(values[0], values[1])
    return LowLevelCallable(wrapped.ctypes)

def jit_integrand_function2(integrand_function):
    jitted_function = numba.jit(integrand_function, nopython=True)
    
    @cfunc(float64(intc, CPointer(float64)))
    def wrapped(n, xx):
        if n < 5:
            raise ValueError("Not enough parameters passed to function.")
        values = carray(xx, n)  # 전달된 모든 파라미터를 포함하는 배열
        return jitted_function(values[0], values[1], values[2], values[3],values[4])
    
    return LowLevelCallable(wrapped.ctypes)



@jit_integrand_function
def integrand_l(x, X):
    return np.cos(x)**2*g(X*np.sin(x),0,1)
@jit_integrand_function
def integrand_d(x,X):
    return (1/x**2)*g(x,0,1)

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
    return g_sec(x,O_pm-shift_O,O_pm)*np.sqrt(1-(x/X)**2)

@jit_integrand_function2
def integrand_Rd2(x, X,O_pm,shift_O,m):
    return 1/(2*(m*(x+O_pm-shift_O)**2+1/m))*g_sec(x,O_pm-shift_O,O_pm)


def FX_0(X,m,O_0):
    O_d = min(O_0,X)
    theta_0 = np.arcsin(O_d/X)
    I_l,err_l = quad(integrand_l,-theta_0,theta_0,limit=200) #lock
    I_d,err_d = quad(integrand_d,O_d,np.inf,limit=200) #drift
    Fl = 1*I_l
    Fd = - 1/(m)*I_d
    F0 = Fl + Fd
    return F0,Fl,Fd
get_FX_0 = np.vectorize(FX_0)

def FX_Rld(X,m,O_0):
    a = 1/np.sqrt(X*m)
    b = 4/np.pi * a - 0.3056*a**3
    b = np.where(np.where(a>1.193,1,b)>=1,1,b)

    omega_p = b*X
    
    R_l,err = quad(integrand_Rl, -omega_p,omega_p,args=(X,O_0,1,m),limit=200)
    R_dr,err = quad(integrand_Rd,omega_p,np.inf,args=(X,O_0,1,m),limit=200)
    R_dl,err = quad(integrand_Rd,-np.inf,-omega_p,args=(X,O_0,1,m),limit=200)

    R_d = R_dr +  R_dl
    R = R_l - R_d
    return R_l,-R_d,R

get_FX_Rld = np.vectorize(FX_Rld)

def Make_R_function(m,O_0,K_max=15):
    X = np.linspace((0.01),(20),num=3000)
    RX_l1,RX_d1,RX_F = get_FX_Rld(X,m,O_0)
    R = RX_l1 + RX_d1
    IK = np.nanmax((R/X))
    Kb =1/IK
    rb = X[np.nanargmax((R/X))]*IK

    Ks = np.linspace(0.5,K_max+0.1,50000)
    A = np.where(np.abs(RX_F*Ks.reshape(-1,1)-X)<5e-5)
    RR = RX_F*np.ones_like(Ks.reshape(-1,1))
    KK = np.ones_like(RX_F)*Ks.reshape(-1,1)

    RR_0 = RX_l1*np.ones_like(Ks.reshape(-1,1))
    U,= np.where(RR[A] >= rb)
    D,= np.where(RR[A] < rb)
    Ku = KK[A][U]
    Kd = KK[A][D]
    Ku,Ku_idx,c=np.unique(Ku,return_counts=True,return_index=True)
    Kd,Kd_idx,c=np.unique(Kd,return_counts=True,return_index=True)

    F_RMu = interpolate.interp1d(KK[A][U][Ku_idx], RR[A][U][Ku_idx], kind='quadratic',bounds_error=False)
    F_R0u = interpolate.interp1d(KK[A][U][Ku_idx], RR_0[A][U][Ku_idx], kind='quadratic',bounds_error=False)
    if m == 0:
        F_RMd = F_RMu
        F_R0d = F_R0u
    else:
        F_RMd = interpolate.interp1d(KK[A][D][Kd_idx], RR[A][D][Kd_idx], kind='quadratic',bounds_error=False)
        F_R0d = interpolate.interp1d(KK[A][D][Kd_idx], RR_0[A][D][Kd_idx], kind='quadratic',bounds_error=False)

    return F_RMd,F_RMu,F_R0d,F_R0u



def F_sec(r,K,m,F_R0,F_RM,O_pm=None, g=g_sec):
    X = K*r
    a = 1/np.sqrt(X*m)
    b = 4/np.pi * a - 0.3056*a**3
    bs = np.where(np.where(a>1.193,1,b)>=1,1,b)

    r_0 = F_R0(K)
    if O_pm is None:
        O_pm = 4/np.pi*np.sqrt(K*F_RM(K)/m) - 0.3056/np.sqrt(K*F_RM(K)*m**3)
        #F_RM으로 부터 얻는 곡선, 만약 shift가 있는경우 그 값에서 부터 시작됨
    # shift_O = -(K**2*r*r_0)/(2*m*(1/m**2+(O_pm)**2)) -(K**2*r*r)/(2*m*(1/m**2+(2*O_pm)**2))
    shift_O = -(K**2*r*r_0)/(2*m*(1/m**2+(O_pm)**2)) + (K**2*r*r)/(2*m**2*O_pm*(1/m**2+(2*O_pm)**2))
    omega_p = bs*X
    O_min = -shift_O - omega_p
    if O_min<0:
        m_ = O_min
    else:
        m_ = np.nan
    I_l,err = quad(integrand_Rl2, shift_O,+omega_p,args=(X,O_pm,shift_O,m),limit=200)
    I_d,err = quad(integrand_Rd2, omega_p,np.inf,args=(X,O_pm,shift_O,m),limit=200)
    F_l = I_l/X
    F_d = -I_d
    F = F_l +  F_d
    return F_l,F_d,F,m_
get_F2 = np.vectorize(F_sec)



def get_r_sec(K,m,FR0,FRM,O_pm=None,samples=200,g_sec=g_sec):
    r0_ =  FR0(K)
    r_sd,r_su = np.nan,np.nan
    r_su_d,r_su_l = np.nan,np.nan
    mu = np.nan
    md = np.nan
    if (K == 0)or (m==0):
        return r_sd,r_su,r_su_d,r_su_l,md,mu
    r_test = np.linspace(1e-5,(1-r0_)/2,samples)
    F_l2,F_d2,F2,m_ = get_F2(r_test,K,m,FR0,FRM,O_pm)
    R2_interpolate  = interpolate.interp1d(r_test,F2, kind='linear',bounds_error=False)
    F2l_interpolate  = interpolate.interp1d(r_test,F_l2, kind='linear',bounds_error=False)
    m_interpolate  = interpolate.interp1d(r_test,m_, kind='linear',bounds_error=False)
    r_test2 = np.linspace(1e-5,(1-r0_)/2,5000)
    Fs = R2_interpolate(r_test2)
    cross_point = np.sign((Fs[0:-1]-1/K)*(Fs[1:]-1/K))*(-0.5) + 0.5
    arg_check, = np.where(cross_point)

    r_sec = (r_test2[arg_check] +r_test2[arg_check+1])/2
    if len(r_sec)==2:
        r_sd,r_su = r_sec
    if len(r_sec)==1:
        r_su = r_sec
    f_su = F2l_interpolate(r_su)
    f_sd = F2l_interpolate(r_sd)
    mu = m_interpolate(r_su)
    md = m_interpolate(r_sd)
    r_su_l = r_su*K*f_su
    r_sd_l = r_sd*K*f_sd
    return r_sd,r_su,r_sd_l,r_su_l,md,mu


# def get_r_sec(K,m,FR0,FRM,O_pm=None,samples=200,g_sec=g_sec):
#     r0_ =  FR0(K)
#     r_sd,r_su = np.nan,np.nan
#     r_su_d,r_su_l = np.nan,np.nan
#     mu = np.nan
#     md = np.nan
#     if (K == 0)or (m==0):
#         return r_sd,r_su,r_su_d,r_su_l,md,mu
#     r_test = np.linspace(1e-5,(1-r0_)/2,samples)
#     F2,m_ = get_F2(r_test,K,m,FR0,FRM,O_pm)
#     R2_interpolate  = interpolate.interp1d(r_test,F2, kind='linear',bounds_error=False)
#     r_test2 = np.linspace(1e-5,(1-r0_)/2,5000)
#     Fs = R2_interpolate(r_test2)
#     cross_point = np.sign((Fs[0:-1]-1/K)*(Fs[1:]-1/K))*(-0.5) + 0.5
#     arg_check, = np.where(cross_point)

#     r_sec = (r_test2[arg_check] +r_test2[arg_check+1])/2
#     if len(r_sec)==2:
#         r_sd,r_su = r_sec
#     if len(r_sec)==1:
#         r_su = r_sec
#     f_su,mu = F_lock2(r_su,K,m,F_R0=FR0,F_RM=FRM,g=g_sec)
#     f_sd,md = F_lock2(r_sd,K,m,F_R0=FR0,F_RM=FRM,g=g_sec)
#     r_su_l = r_su*K*f_su
#     r_sd_l = r_sd*K*f_sd
#     return r_sd,r_su,r_sd_l,r_su_l,md,mu

get_r_sec_np = np.vectorize(get_r_sec)


def get_r_rp(m,O_0 = 0,K_max = 10,K_len = 201):
    F_r = {}
    F_RMd,F_RMu,F_R0d,F_R0u = Make_R_function(m,O_0,K_max=K_max)
    F_r['R_u'] = F_RMu
    F_r['R_d'] = F_RMd
    F_r['R0_u'] = F_R0u
    F_r['R0_d'] = F_R0d
    Ks_ = np.linspace(0,K_max,K_len)
    r_sd,r_su,r_sd_l,r_su_l,md,mu = get_r_sec_np(Ks_,m,F_R0u,F_RMu)

    F_r['Ks'] = Ks_
    F_r['r_+u'] = r_su
    F_r['r_+d'] = r_sd
    F_r['r_+0u'] = r_su_l
    F_r['r_+0d'] = r_sd_l
    F_r['F_md'] = md
    F_r['F_mu'] = mu
    return F_r



