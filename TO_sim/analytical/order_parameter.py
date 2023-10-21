from scipy.integrate import quad
import scipy.stats as SS
import numpy as np
import matplotlib.pyplot as plt
import parmap
import sympy as sym
import multiprocessing as mp
from scipy.interpolate import interp1d

core = mp.cpu_count()

def g_c(x):
    return SS.cauchy.pdf(x,0,1)
def g_n(x):
    return SS.norm.pdf(x,0,1)

def dK_graph(data,dK):
    ln_x , ln_y = map(np.array,data)
    ln_Ks = np.arange(ln_x[0],ln_x[-1],dK)
    ln_Kidx = np.searchsorted(ln_x,ln_Ks)
    ln_r = ln_y[ln_Kidx]
    return ln_Ks,ln_r

def split_uo(K,R):
    K,R = map(np.array,[K,R])
    min_KR = R[np.argmin(K)]
    cond_over, = np.where(R>=min_KR)
    cond_under, = np.where(R<min_KR)
    R_over = R[cond_over]
    R_under = R[cond_under]

    K_over = K[cond_over]
    K_under = K[cond_under]
    K_under,R_under = dK_graph((K_under,R_under),0.05)
    K_over,R_over = dK_graph((K_over,R_over),0.05)
    return (K_under,R_under),(K_over,R_over)

def r_lock1(X,m,g):
    integrand_lock = lambda x:np.cos(x)**2*g(X*np.sin(x))
    omega_p = (4/np.pi)*np.sqrt(X/m)

    A = omega_p/X
    if abs(A)<=1:
        theta_p = np.arcsin(A)
        I_l,err = quad(integrand_lock,-theta_p,theta_p,limit=200)
        return X*I_l

    else: return np.NaN

def r_drift1(X,m,g):
    O_p = (4/np.pi)*np.sqrt(X/m)
    integrand_drift = lambda x:1/(x**2)*g(x)
    I_d,err = quad(integrand_drift,O_p,np.inf,limit=200)
    return -X/(m)*I_d
def r_1(X,m,g):
    integrand_lock = lambda x:np.cos(x)**2*g(X*np.sin(x))
    O_p = (4/np.pi)*np.sqrt(X/m)
    A = O_p/X
    integrand_drift = lambda x:1/(x**2)*g(x)
    I_d,err = quad(integrand_drift,O_p,np.inf,limit=200)
    r_d = -X/(m)*I_d

    if abs(A)<=1:
        theta_p = np.arcsin(A)
        I_l,err = quad(integrand_lock,-theta_p,theta_p,limit=200)
        r_l = X*I_l
    else: r_l = 0 #np.NaN
    return r_l + r_d

def r_lock2(X,m,g):
    integrand_lock = lambda x:np.cos(x)**2*g(X*np.sin(x))
    I_l,err = quad(integrand_lock,-np.pi/2,np.pi/2,limit=200)
    return X*I_l


def r_drift2(X,m,g):
    O_d = X
    integrand_drift = lambda x:1/(x**2)*g(x)
    I_d,err = quad(integrand_drift,O_d,np.inf,limit=200)
    return -X/(m)*I_d

def r_2(X,m,g):
    O_d = X
    integrand_lock = lambda x:np.cos(x)**2*g(X*np.sin(x))
    integrand_drift = lambda x:1/(x**2)*g(x)
    I_l,err = quad(integrand_lock,-np.pi/2,np.pi/2,limit=200)
    I_d,err = quad(integrand_drift,O_d,np.inf,limit=200)
    r = X*I_l -X/(m)*I_d
    return r

def r_0(X,m,g,O_0):
        O_d = min(O_0,X)
        theta_0 = np.arcsin(O_d/X)
        integrand_l = lambda x:np.cos(x)**2*g(X*np.sin(x))
        integrand_d = lambda x:(1/x**2)*g(x)
        I_l,err_l = quad(integrand_l,-theta_0,theta_0,limit=200) #lock
        I_d,err_d = quad(integrand_d,O_d,np.inf,limit=200) #drift
        r0 = X*I_l - X/(m)*I_d
        return r0

r_1 = np.vectorize(r_1)
r_2 = np.vectorize(r_2)


r_drift1 = np.vectorize(r_drift1)
r_drift2 = np.vectorize(r_drift2)
r_lock1 = np.vectorize(r_lock1)
r_lock2 = np.vectorize(r_lock2)

def Make_empirical_KR(m,dist='normal'):
    if dist.upper() == "Normal".upper():
        gen_dist = g_n
    else:
        gen_dist = g_c
    X = np.logspace(np.log10(0.1),np.log10(50),num=1000,base=10)
    Ks = np.linspace(0.01,30,20000)
    if m != 0:
        r_l1=r_lock1(X,m=m,g=gen_dist)
        r_l2=r_lock2(X,m=m,g=gen_dist)
        r_d1=r_drift1(X,m=m,g=gen_dist)
        r_d2=r_drift2(X,m=m,g=gen_dist)
        r_l2,r_d2,r_l1,r_d1 = map(np.array,[r_l2,r_d2,r_l1,r_d1])
    else:
        r_l1=r_lock1(X,m=m,g=gen_dist)
        r_l2=r_lock2(X,m=m,g=gen_dist)
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

def Make_empirical_KR_0(m,O_0,dist='normal'):
    X = np.logspace(np.log10(0.1),np.log10(50),num=1000,base=10)
    Ks = np.linspace(0.01,20,20000)
    if dist.upper() == "Normal".upper():
        g= g_n
    else:
        g= g_c
    def r_0(X,m,g,O_0):
        O_d = min(O_0,X)
        theta_0 = np.arcsin(O_d/X)
        integrand_l = lambda x:np.cos(x)**2*g(X*np.sin(x))
        integrand_d = lambda x:(1/x**2)*g(x)
        I_l,err_l = quad(integrand_l,-theta_0,theta_0,limit=200) #lock
        I_d,err_d = quad(integrand_d,O_d,np.inf,limit=200) #drift
        r0 = X*I_l - X/(m)*I_d
        return r0
    # def r_0(X,m,g,O_0):
    #     O_d = min(O_0,X)
    #     theta_0 = np.arcsin(O_d/X)
    #     integrand_l = lambda x:np.cos(x)**2*g(X*np.sin(x))
    #     integrand_l = lambda x:g(x)*(1-x**2/X**2)**0.5

    #     integrand_d = lambda x,m=m:(X*m/(2*(1+(x*m)**2)))*g(x)
    #     I_l,err_l = quad(integrand_l,0,O_d,limit=200) #lock
    #     I_d,err_d = quad(integrand_d,O_d,np.inf,limit=200) #drift
    #     r0 = 2*I_l - 2*I_d
    #     return r0
    r_0 = np.vectorize(r_0)
    

    r_before = r_0(X,m,g,O_0)
    KB =[]
    RB =[]
    for K_  in  Ks:
        TEMP_2 = (X/K_)[abs(r_before-X/K_)<5e-4]
        if len(TEMP_2)!=0:
            for R_ in TEMP_2:
                KB.append(K_)
                RB.append(R_)
    (K_under,R_under),(K_over,R_over) = split_uo(KB,RB)
    return (K_under,R_under),(K_over,R_over)


