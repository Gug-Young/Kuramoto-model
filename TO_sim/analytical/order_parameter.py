from scipy.integrate import quad
import scipy.stats as SS
import numpy as np
import matplotlib.pyplot as plt
import parmap
import sympy as sym
import multiprocessing as mp

core = mp.cpu_count()


def g_c(x):
    return SS.cauchy.pdf(x,0,1)
def g_n(x):
    return SS.norm.pdf(x,0,1)


def r_lock1(X,m,g):
    integrand = lambda x:np.cos(x)**2*g(X*np.sin(x))

    omega_p = (4/np.pi)*np.sqrt(X/m)
    A = omega_p/X
    if abs(A)<=1:
        theta_p = np.arcsin(A)
        I_ = quad(integrand,-theta_p,theta_p,limit=200)
        return X*I_[0]

    else: return np.NaN


def r_lock2(X,m,g):
    integrand = lambda x:np.cos(x)**2*g(X*np.sin(x))
    I_ = quad(integrand,-np.pi/2,np.pi/2,limit=200)
    return X*I_[0]

def r_drift2(X,m,g):
    O_d = X
    A = lambda x:1/(x**3)
    integrand = lambda x:A(x)*g(x)
    I_ = quad(integrand,O_d,np.inf,limit=200)
    return -X/(m**2)*I_[0]

def r_drift1(X,m,g):
    O_p = (4/np.pi)*np.sqrt(X/m)
    A = lambda x:1/(x**3)
    integrand = lambda x:A(x)*g(x)
    I_ = quad(integrand,O_p,np.inf,limit=200)
    return -X/(m**2)*I_[0]


def Make_empirical_KR(m,dist='normal'):
    if dist.upper() == "Normal".upper():
        gen_dist = g_n
    else:
        gen_dist = g_c
    X = np.logspace(np.log10(0.1),np.log10(50),num=1000,base=10)
    Ks = np.linspace(0.01,13,20000)
    if m != 0:
        r_l1=parmap.map(r_lock1,X,m=m,g=gen_dist,pm_processes=core,pm_pbar=False)
        r_l2=parmap.map(r_lock2,X,m=m,g=gen_dist,pm_processes=core,pm_pbar=False)
        r_d1=parmap.map(r_drift1,X,m=m,g=gen_dist,pm_processes=core,pm_pbar=False)
        r_d2=parmap.map(r_drift2,X,m=m,g=gen_dist,pm_processes=core,pm_pbar=False)
        r_l2,r_d2,r_l1,r_d1 = map(np.array,[r_l2,r_d2,r_l1,r_d1])
    else:
        r_l1=parmap.map(r_lock1,X,m=m,g=gen_dist,pm_processes=core,pm_pbar=False)
        r_l2=parmap.map(r_lock2,X,m=m,g=gen_dist,pm_processes=core,pm_pbar=False)
        r_l2,r_l1 = map(np.array,[r_l2,r_l1])
        
        r_d1=0*r_l1
        r_d2=0*r_l2
        
    r_case1 = r_l1+r_d1
    r_case2 = r_l2+r_d2

    KF =[]
    RF =[]
    KB =[]
    RB =[]

    for K_  in  Ks:
            i=0
            # plt.plot(X,X/K_)
            i+=1
            TEMP_2 = (X/K_)[abs(r_case2-X/K_)<1e-3]
            if len(TEMP_2)!=0:
                for R_ in TEMP_2:
                    KB.append(K_)
                    RB.append(R_)
            TEMP_1 = (X/K_)[abs(r_case1-X/K_)<1e-3]
            if len(TEMP_1)!=0:
                for R_ in TEMP_1:
                    KF.append(K_)
                    RF.append(R_)
    KF,RF,KB,RB = map(np.array,[KF,RF,KB,RB])          
    return KF,RF,KB,RB