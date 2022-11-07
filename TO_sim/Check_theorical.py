import numpy as np
import matplotlib.pyplot as plt

def Check_min_omega(m):
    Omega_D = lambda K,r : K*r
    Omega_P = lambda K,r,m : (4/np.pi)*np.sqrt((K*r)/m)
    Kspace = np.linspace(0.01,12.5,1000)
    Rspace = np.linspace(0.01,1,1000)
    KS,RS = np.meshgrid(Kspace,Rspace)
    OP = Omega_P(KS,RS,m)
    OD = Omega_D(KS,RS)
    MA = np.ma.masked_where(OD <= OP, OP)
    min_omega = MA.min()
    return min_omega

def K_R_Lorentzian_theorical(K,Kc):
    R = lambda k: np.sqrt(1-Kc/k)
    r = np.array([R(x) if x>Kc else 0 for x in K])
    return r
def Make_theorical_KR(Ks,m,d=1):
    def r_Case1(x,m,d):
        O_p = (4/np.pi)*np.sqrt(x/m)
        t_p = np.arcsin(O_p/x)
        pi = np.pi
        r_lock1 = (x/m**2)*((1/(pi*d**3))*np.log(np.sqrt(d**2+O_p**2)/O_p)-1/(2*pi*d*O_p**2))
        r_drift1 = (2/(pi*x))*(np.sqrt(d**2+x**2)*np.arctan(np.sqrt(d**2+x**2)*np.tan(t_p))-d*t_p)
        return r_lock1+r_drift1
    r_Case2 = lambda x,m,d:(x/(m**2))*((1/np.pi*d**3)*np.log(np.sqrt(d**2+x**2)/(x))-1/(2*np.pi*d*x**2))+(np.sqrt(d**2+x**2)-d)/x
    X = np.logspace(np.log10(0.001),np.log10(500),num=10000,base=10)
    
    KF =[]
    RF =[]
    KB =[]
    RB =[]

    for K_  in  Ks[::1]:
            i=0
            # plt.plot(X,X/K_)
            i+=1
            TEMP_2 = (X/K_)[abs(r_Case2(X,m,d)-X/K_)<1e-3]
            if len(TEMP_2)!=0:
                for R_ in TEMP_2:
                    KB.append(K_)
                    RB.append(R_)
            TEMP_1 = (X/K_)[abs(r_Case1(X,m,d)-X/K_)<1e-3]
            if len(TEMP_1)!=0:
                for R_ in TEMP_1:
                    KF.append(K_)
                    RF.append(R_)
    return KF,RF,KB,RB

