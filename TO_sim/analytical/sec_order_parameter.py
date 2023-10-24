
import numpy as np
import pandas as pd
from TO_sim.gen_Distribution import Normal, Quantile_Normal as Q_Normal, Lorentzian
from scipy.integrate import quad
from scipy.stats import norm




def Bisection(f,r_a,r_b,eps =1e-5,end=30,arg=()):
    r_c = (r_a+r_b)/2
    num = 0 
    
    while abs(f(r_c,*arg))>eps:
        r_c = (r_a+r_b)/2
        if f(r_a,*arg)*f(r_c,*arg)>0:
            r_a = r_c
        else:
            r_b = r_c
        num+=1
        if num >end:
            # print('Please select another section')
            return np.NAN
    # print(f'count : {num}')
    return r_c

def g_n(x):
    return norm.pdf(x,0,1)

def r_lock1(r,K,m,g=g_n):
    X = K*r
    integrand_lock = lambda x:np.cos(x)**2*g(X*np.sin(x))
    omega_p = (4/np.pi)*np.sqrt(X/m)

    A = omega_p/X
    if abs(A)<=1:
        theta_p = np.arcsin(A)
        I_l,err = quad(integrand_lock,-theta_p,theta_p,limit=200)
        return X*I_l

    else: 
        theta_p = np.arcsin(A)
        I_l,err = quad(integrand_lock,-np.pi/2,np.pi/2,limit=200)
        return X*I_l
    

def r_drift1(r,K,m,g=g_n):
    X = K*r
    O_p = (4/np.pi)*np.sqrt(X/m)
    integrand_drift = lambda x:1/(x**2)*g(x)
    I_d,err = quad(integrand_drift,O_p,np.inf,limit=200)
    return -X/(m)*I_d


def r_mean(r,K,m,g=g_n):
    rl = r_lock1(r,K,m,g=g_n)
    rd = r_drift1(r,K,m,g=g_n)
    return rl+rd - r

def F_lock1(r,K,m,g=g_n):
    X = K*r
    integrand_lock = lambda x:np.cos(x)**2*g(X*np.sin(x))
    omega_p = (4/np.pi)*np.sqrt(X/m)

    A = omega_p/X
    if abs(A)<=1:
        theta_p = np.arcsin(A)
        I_l,err = quad(integrand_lock,-theta_p,theta_p,limit=200)
        return I_l

    else: 
        theta_p = np.arcsin(A)
        I_l,err = quad(integrand_lock,-np.pi/2,np.pi/2,limit=200)
        return I_l
def F_drift1(r,K,m,g=g_n):
    X = K*r
    O_p = (4/np.pi)*np.sqrt(X/m)
    integrand_drift = lambda x:1/(x**2)*g(x)
    I_d,err = quad(integrand_drift,O_p,np.inf,limit=200)
    return -1/(m)*I_d

def F_lock1b(r,K,m,g=g_n):
    X = K*r
    integrand_lock = lambda x:np.cos(x)**2*g(X*np.sin(x))
    omega_p = (4/np.pi)*np.sqrt(X/m)
    I_l,err = quad(integrand_lock,-np.pi/2,np.pi/2,limit=200)
    return I_l
def F_drift1b(r,K,m,g=g_n):
    X = K*r
    O_d = X
    integrand_drift = lambda x:1/(x**2)*g(x)
    I_d,err = quad(integrand_drift,O_d,np.inf,limit=200)
    return -1/(m)*I_d

F_l1 = np.vectorize(F_lock1)
F_d1 = np.vectorize(F_drift1)

F_l1b = np.vectorize(F_lock1b)
F_d1b = np.vectorize(F_drift1b)

def F_mean(r,K,m,g=g_n):
    Fl = F_l1(r,K,m,g=g)
    Fd = F_d1(r,K,m,g=g)
    return Fl+Fd

def F_mean_b(r,K,m,g=g_n):
    Fl = F_l1b(r,K,m,g=g)
    Fd = F_d1b(r,K,m,g=g)
    return Fl+Fd


def get_r_mean(K,m,g=g_n,samples=200):
    rm_u = np.nan
    rm_d = np.nan
    r_1test = np.linspace(1e-4,1,samples)
    F1  = F_mean(r_1test,K,m,g=g)
    cross_point = np.sign((F1[0:-1]-1/K)*(F1[1:]-1/K))*(-0.5) + 0.5
    arg_check, = np.where(cross_point)
    near_check = np.argmin(abs(F1[:]-1/K))
    check_err = abs(F1[near_check] - 1/K)<1e-3
    rm_u = np.nan
    rm_d = np.nan
    if len(arg_check)==2:
        r_ss = []
        for arg in arg_check:
            r_a = r_1test[arg]
            r_b = r_1test[arg+1]
            r_m = Bisection(r_mean,r_a,r_b,eps=1e-3,end=15,arg = (K,m,g))
            if np.isnan(r_m):
                r_m = (r_a+r_b)/2
            r_ss.append(r_m)
        rm_d,rm_u = np.sort(r_ss)
        return rm_d,rm_u
    elif len(arg_check)==1:
        for arg in arg_check:
            r_a = r_1test[arg]
            r_b = r_1test[arg+1]
            r_m = Bisection(r_mean,r_a,r_b,eps=1e-3,end=15,arg = (K,m,g))
            if np.isnan(r_m):
                r_m = (r_a+r_b)/2
        rm_u = r_m
        return rm_d,rm_u
    else:
        if check_err:
            r_a = r_1test[near_check] - 0.1
            r_b = r_1test[near_check] + 0.1
            r_m = Bisection(r_mean,r_a,r_b,eps=1e-3,end=15,arg = (K,m,g))
            print(r_m)
            rm_u = r_m
        return rm_d,rm_u
    

def get_r_mean_b(K,m,g=g_n,samples=200):
    rm_u = np.nan
    rm_d = np.nan
    r_1test = np.linspace(1e-4,1,samples)
    F1  = F_mean_b(r_1test,K,m,g=g)
    cross_point = np.sign((F1[0:-1]-1/K)*(F1[1:]-1/K))*(-0.5) + 0.5
    arg_check, = np.where(cross_point)
    near_check = np.argmin(abs(F1[:]-1/K))
    check_err = abs(F1[near_check] - 1/K)<1e-3
    rm_u = np.nan
    rm_d = np.nan
    if len(arg_check)==2:
        r_ss = []
        for arg in arg_check:
            r_a = r_1test[arg]
            r_b = r_1test[arg+1]
            r_m = Bisection(r_mean,r_a,r_b,eps=1e-3,end=15,arg = (K,m,g))
            if np.isnan(r_m):
                r_m = (r_a+r_b)/2
            r_ss.append(r_m)
        rm_d,rm_u = np.sort(r_ss)
        return rm_d,rm_u
    elif len(arg_check)==1:
        for arg in arg_check:
            r_a = r_1test[arg]
            r_b = r_1test[arg+1]
            r_m = Bisection(r_mean,r_a,r_b,eps=1e-3,end=15,arg = (K,m,g))
            if np.isnan(r_m):
                r_m = (r_a+r_b)/2
        rm_u = r_m
        return rm_d,rm_u
    else:
        if check_err:
            r_a = r_1test[near_check] - 0.1
            r_b = r_1test[near_check] + 0.1
            r_m = Bisection(r_mean,r_a,r_b,eps=1e-3,end=15,arg = (K,m,g))
            print(r_m)
            rm_u = r_m
        return rm_d,rm_u
rm_numpy = np.vectorize(get_r_mean)
rm_b_numpy = np.vectorize(get_r_mean_b)



def g_sec(x,Or,Om):
    g = norm.pdf(x,-Or,1)
    dO = abs(Or-Om)
    return np.where(x<-dO,1e-6,g)
def r_lock2(r,r_m,O_r,O_pm,K,m,g=g_sec):
    X = K*r
    shift = lambda x,X=X: -(K**2*r*r_m)/(2*(O_pm+X*np.sin(x))**2)/m -(K**2*r*r)/(8*(O_pm+X*np.sin(x))**2)/m 

    integrand_lock = lambda x:np.cos(x)**2*g(X*np.sin(x)+shift(0),O_r,O_pm)
    omega_p = (4/np.pi)*np.sqrt(X/m)

    A = omega_p/X
    if abs(A)<=1:
        theta_p = np.arcsin(A)
        I_l,err = quad(integrand_lock,-theta_p,theta_p,epsabs=1e-10,limit=2000)
        return X*I_l

    else: 
        I_l,err = quad(integrand_lock,-np.pi/2,np.pi/2,epsabs=1e-10,limit=2000)
        return X*I_l    

def r_sec(r,r_m,O_r,O_pm,K,m,g=g_sec):
    rl = r_lock2(r,r_m,O_r,O_pm,K,m,g=g_sec)
    rd = 0#r_drift1(r,K,m,g=g_n)
    return rl+rd - r

def F_lock2(r,r_m,O_r,O_pm,K,m,g=g_sec):
    X = K*r
    shift = lambda x,X=X: -(K**2*r*r_m)/(2*(O_pm+X*np.sin(x))**2)/m -(K**2*r*r)/(8*(O_pm+X*np.sin(x))**2)/m 

    integrand_lock = lambda x:np.cos(x)**2*g(X*np.sin(x)+shift(0),O_r,O_pm)
    omega_p = (4/np.pi)*np.sqrt(X/m)

    A = omega_p/X
    if abs(A)<=1:
        theta_p = np.arcsin(A)
        I_l,err = quad(integrand_lock,-theta_p,theta_p,epsabs=1e-10,limit=2000)
        return I_l

    else: 
        I_l,err = quad(integrand_lock,-np.pi/2,np.pi/2,epsabs=1e-10,limit=2000)
        return I_l    
    
def get_F_sec(r,K,m,r_0):
    O_pm = norm.ppf(r_0/2+0.5)
    O_r = quad(norm.ppf,r_0/2+0.5,1)[0]/(0.5-r_0/2)
    F_2 = F_lock2(r,r_0,O_r,O_pm,K,m,g_sec)
    return F_2
F_sec = np.vectorize(get_F_sec)

def get_r_sec(K,m,r_0,samples=200):
    rs_u = np.nan
    rs_d = np.nan
    if np.isnan(r_0):
        return rs_u,rs_d
    r_2test = np.linspace(0,(1-r_0)/2,samples)
    O_pm = norm.ppf(r_0/2+0.5)
    O_r = quad(norm.ppf,r_0/2+0.5,1)[0]/(0.5-r_0/2)
    Fs  = F_sec(r_2test,K,m,r_0)
    cross_point = np.sign((Fs[0:-1]-1/K)*(Fs[1:]-1/K))*(-0.5) + 0.5
    arg_check, = np.where(cross_point)
    near_check = np.argmin(abs(Fs[:]-1/K))
    check_err = abs(Fs[near_check] - 1/K)<1e-3
    rs_u = np.nan
    rs_d = np.nan
    if len(arg_check)==2:
        r_ss = []
        for arg in arg_check:
            r_a = r_2test[arg]
            r_b = r_2test[arg+1]
            r_s = Bisection(r_sec,r_a,r_b,eps=5e-3,end=15,arg = (r_0,O_r,O_pm,K,m,g_sec))
            if np.isnan(r_s):
                r_s = (r_a+r_b)/2
            r_ss.append(r_s)
        rs_d,rs_u = r_ss
        return rs_d,rs_u
    elif len(arg_check)==1:
        for arg in arg_check:
            r_a = r_2test[arg]
            r_b = r_2test[arg+1]
            r_s = Bisection(r_sec,r_a,r_b,eps=5e-3,end=15,arg = (r_0,O_r,O_pm,K,m,g_sec))
            if np.isnan(r_s):
                r_s = (r_a+r_b)/2
        rs_u = r_s
        return rs_d,rs_u
    else:
        if check_err:
            r_a = r_2test[near_check] - 0.1
            r_b = r_2test[near_check] + 0.1
            r_s = Bisection(r_sec,r_a,r_b,eps=5e-3,end=15,arg = (r_0,O_r,O_pm,K,m,g_sec))
            print(r_s)
            rs_u = r_s
        return rs_d,rs_u


rm_numpy = np.vectorize(get_r_mean)
rs_numpy = np.vectorize(get_r_sec)
# F_m1 = np.vectorize(F_mean)

import warnings
warnings.filterwarnings(action='ignore')

def make_r_rsec(m,Ks):
    r_d,r_u  = rm_numpy(Ks,m)
    r_mu = F_l1(r_u,Ks,m)*(Ks*r_u)
    rs_d,rs_u = rs_numpy(Ks,m,r_u)
    return m,Ks,r_d,r_u,r_mu,rs_d,rs_u
