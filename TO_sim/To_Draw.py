import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from TO_sim.Check_theoretical import Make_theoretical_KR


def Time_R(Ks,t_dic,rs_dic,t_r_dic,rs_r_dic,dK,dt,t_end,N,m,mean_time=50,save=False,dist="Quantile Lorentzian",Folder_name="Review",K_draw=(1,13),r_draw=(0,0.9)):
    int_ =np.linspace(0.0,1,len(Ks))
    color = plt.cm.viridis_r(int_)
    fig, (ax1,ax2) = plt.subplots(2,1)
    t_ = t_dic[0]
    num = 0
    for t_temp, r_temp in zip([*t_dic.values()],[*rs_dic.values()]):
        ax1.plot(t_temp,r_temp,color = color[num],label=f"{Ks[num]:.02f}")
        ax1.vlines(t_temp[0],-0.01,1.1,alpha=0.5,linestyles='--',color=color[num])
        if num%10==0:
            text_ = ax1.text(t_temp[0],r_temp[-1]+0.2,f"K = {Ks[num]:.01f}",fontsize=8)
            text_.set_bbox(dict(facecolor=color[num], alpha=0.5, edgecolor=color[num]))
        num +=1
    for t_temp, r_temp in zip([*t_r_dic.values()],[*rs_r_dic.values()]):
        num -=1
        ax2.vlines(t_temp[0],-0.01,1.1,alpha=0.5,linestyles='--',color=color[num])
        if num%10==0:
            text_ = ax2.text(t_temp[-1],r_temp[0]+0.2,f"K = {Ks[num]:.01f}",fontsize=8)
            text_.set_bbox(dict(facecolor=color[num], alpha=0.5, edgecolor=color[num]))
        ax2.plot(t_temp,r_temp,color = color[num])
    ax2.invert_xaxis()
    ax1.set_ylabel("Order parameter")
    ax2.set_ylabel("Order parameter")
    ax1.set_xlabel(r"time [s] ($0 \rightarrow \infty$)")
    ax2.set_xlabel("time [s] ($\infty \leftarrow 0$)")
    # ax1.legend(ncol=6,fontsize=3,loc='center left', bbox_to_anchor=(1, 0))
    # plt.colorbar()
    ax1.text(0,0.7,"Forward")
    ax2.text(t_end*len(Ks),0.7,"Backward")
    plt.tight_layout()
    if save:
        plt.savefig(f'{Folder_name}/N = {N}, m = {m},Forward,Backward Full time,t_end={t_end},{dist},{dK}.png',dpi=400)
    
    
    fig, (ax1,ax2) = plt.subplots(2,1)
    
    t_ = t_dic[0]
    
    num = 0
    for t_temp, r_temp in zip([*t_dic.values()],[*rs_dic.values()]):
        # if num ==15:
        ax1.plot(t_,r_temp,color = color[num],label=f"K = {Ks[num]:.02f}")
        num +=1
        
    for t_temp, r_temp in zip([*t_r_dic.values()],[*rs_r_dic.values()]):
        num -=1
        # if num ==12:
        ax2.plot(t_,r_temp,color = color[num],label=f"K = {Ks[num]:.02f}")
    ax1.set_ylabel("Order parameter")
    ax2.set_ylabel("Order parameter")
    ax1.set_xlabel(r"time [s]")
    ax2.set_xlabel("time [s]")
    # ax1.legend(ncol=6,fontsize=3,loc='center left', bbox_to_anchor=(1, 0))
    # plt.colorbar()
    text_1 = ax1.text(0,0.7,"Forward")
    text_1.set_bbox(dict(facecolor=color[num], alpha=0.5, edgecolor=color[num]))
    text_2 = ax2.text(0,0.7,"Backward")
    text_2.set_bbox(dict(facecolor=color[num], alpha=0.5, edgecolor=color[num]))
    ax1.grid()
    ax2.grid()
    # ax1.legend(loc=4)
    # ax2.legend(loc=0)
    plt.tight_layout()
    if save:
        plt.savefig(f'{Folder_name}/N = {N}, m = {m},Forward,Backward each time,t_end={t_end},{dist},{dK}.png',dpi=400)
    
    r_last = []
    r_last_std = []
    r_r_last = []
    r_r_last_std = []
    Mean_time = int(mean_time/dt)
    for t_temp, r_temp in zip([*t_dic.values()],[*rs_dic.values()]):
        r_time_temp = r_temp[-Mean_time:]
        r_time_std = np.std(r_time_temp)
        r_last.append(np.mean(r_time_temp))
        r_last_std.append(r_time_std)
        
    for t_temp, r_temp in zip([*t_r_dic.values()],[*rs_r_dic.values()]):
        r_r_time_temp = r_temp[-Mean_time:]
        r_r_time_std = np.std(r_r_time_temp)
        r_r_last.append(np.mean(r_r_time_temp))
        r_r_last_std.append(r_r_time_std)
        
    plt.figure()
    # plt.plot(Ks,r_last,'d',markersize=6)
    plt.errorbar(Ks,r_last,yerr=r_last_std,fmt='d',markersize=6,capsize=3)
    # plt.plot(Ks[::-1],r_r_last,'d',markersize=6)
    plt.errorbar(Ks[::-1],r_r_last,yerr=r_r_last_std,fmt='d',markersize=6,capsize=3)
    
    plt.grid()
    plt.ylim(*r_draw)
    plt.xlim(*K_draw)
    plt.tight_layout()
    plt.ylabel("r (order parameter)",fontsize=13)
    plt.xlabel("K (coupling constant)",fontsize=13)
    plt.title(f"K vs r, m = {m}, N = {N}",fontsize=15)
    plt.tight_layout()
    if save:
        plt.savefig(f"{Folder_name}/N = {N}, m = {m}, k vs r {dist},{dK},{t_end}.png",transparent=True,dpi = 500)
def Time_R_df(Ksdf,Ksrdf,N,m,mean_time=50,save=False,dist="Quantile Lorentzian",Folder_name="Review",K_draw=(1,13),r_draw=(0,0.9)):
    Ks = Ksdf.index
    Ksr = Ksrdf.index
    int_ =np.linspace(0.0,1,len(Ks))
    color = plt.cm.viridis_r(int_)
    fig, (ax1,ax2) = plt.subplots(2,1)
    t_ = Ksdf["ts"].iloc[0]
    dK = Ks[1]-Ks[0]
    t_end = Ksdf["ts"].iloc[0][-1]
    dt = Ksdf["ts"].iloc[0][1]-Ksdf["ts"].iloc[0][0]
    num = 0
    for t_temp, r_temp in zip(Ksdf["ts"],Ksdf["rs"]):
        ax1.plot(t_temp,r_temp,color = color[num],label=f"{Ks[num]:.02f}")
        ax1.vlines(t_temp[0],-0.01,1.1,alpha=0.5,linestyles='--',color=color[num])
        if num%10==0:
            text_ = ax1.text(t_temp[0],r_temp[-1]+0.2,f"K = {Ks[num]:.01f}",fontsize=8)
            text_.set_bbox(dict(facecolor=color[num], alpha=0.5, edgecolor=color[num]))
        num +=1
    for t_temp, r_temp in zip(Ksrdf["ts"],Ksrdf["rs"]):
        num -=1
        ax2.vlines(t_temp[0],-0.01,1.1,alpha=0.5,linestyles='--',color=color[num])
        if num%10==0:
            text_ = ax2.text(t_temp[-1],r_temp[0]+0.2,f"K = {Ks[num]:.01f}",fontsize=8)
            text_.set_bbox(dict(facecolor=color[num], alpha=0.5, edgecolor=color[num]))
        ax2.plot(t_temp,r_temp,color = color[num])
    ax2.invert_xaxis()
    ax1.set_ylabel("Order parameter")
    ax2.set_ylabel("Order parameter")
    ax1.set_xlabel(r"time [s] ($0 \rightarrow \infty$)")
    ax2.set_xlabel("time [s] ($\infty \leftarrow 0$)")
    # ax1.legend(ncol=6,fontsize=3,loc='center left', bbox_to_anchor=(1, 0))
    # plt.colorbar()
    ax1.text(0,0.7,"Forward")
    ax2.text(t_end*len(Ks),0.7,"Backward")
    fig.tight_layout()
    if save:
        plt.savefig(f'{Folder_name}/N = {N}, m = {m},Forward,Backward Full time,t_end={t_end},{dist},{dK}.png',dpi=400)
    
    
    fig, (ax1,ax2) = plt.subplots(2,1)
        
    num = 0
    for t_temp, r_temp in zip(Ksdf["ts"],Ksdf["rs"]):
        # if num ==15:
        ax1.plot(t_,r_temp,color = color[num],label=f"K = {Ks[num]:.02f}")
        num +=1
        
    for t_temp, r_temp in zip(Ksrdf["ts"],Ksrdf["rs"]):
        num -=1
        # if num ==12:
        ax2.plot(t_,r_temp,color = color[num],label=f"K = {Ks[num]:.02f}")
    ax1.set_ylabel("Order parameter")
    ax2.set_ylabel("Order parameter")
    ax1.set_xlabel(r"time [s]")
    ax2.set_xlabel("time [s]")
    # ax1.legend(ncol=6,fontsize=3,loc='center left', bbox_to_anchor=(1, 0))
    # plt.colorbar()
    text_1 = ax1.text(0,0.7,"Forward")
    text_1.set_bbox(dict(facecolor=color[num], alpha=0.5, edgecolor=color[num]))
    text_2 = ax2.text(0,0.7,"Backward")
    text_2.set_bbox(dict(facecolor=color[num], alpha=0.5, edgecolor=color[num]))
    ax1.grid()
    ax2.grid()
    # ax1.legend(loc=4)
    # ax2.legend(loc=0)
    fig.tight_layout()
    if save:
        plt.savefig(f'{Folder_name}/N = {N}, m = {m},Forward,Backward each time,t_end={t_end},{dist},{dK},dt={dt}.png',dpi=400)
    
    r_last = []
    r_last_std = []
    r_r_last = []
    r_r_last_std = []
    Mean_time = int(mean_time/dt)
    for t_temp, r_temp in zip(Ksdf["ts"],Ksdf["rs"]):
        r_time_temp = r_temp[-Mean_time:]
        r_time_std = np.std(r_time_temp)
        r_last.append(np.mean(r_time_temp))
        r_last_std.append(r_time_std)
        
    for t_temp, r_temp in zip(Ksrdf["ts"],Ksrdf["rs"]):
        r_r_time_temp = r_temp[-Mean_time:]
        r_r_time_std = np.std(r_r_time_temp)
        r_r_last.append(np.mean(r_r_time_temp))
        r_r_last_std.append(r_r_time_std)
        
    plt.figure()
    # plt.plot(Ks,r_last,'d',markersize=6)
    plt.errorbar(Ks,r_last,yerr=r_last_std,fmt='d',markersize=6,capsize=3)
    # plt.plot(Ks[::-1],r_r_last,'d',markersize=6)
    plt.errorbar(Ks[::-1],r_r_last,yerr=r_r_last_std,fmt='d',markersize=6,capsize=3)
    
    plt.grid()
    plt.ylim(*r_draw)
    plt.xlim(*K_draw)
    plt.tight_layout()
    plt.ylabel("r (order parameter)",fontsize=13)
    plt.xlabel("K (coupling constant)",fontsize=13)
    plt.title(f"K vs r, m = {m}, N = {N}, dt = {dt}",fontsize=15)
    plt.tight_layout()
    if save:
        plt.savefig(f"{Folder_name}/N = {N}, m = {m}, k vs r {dist},{dK},{t_end},dt={dt}.png",transparent=True,dpi = 500)

def Time_R_df_total(Ksdf,Ksrdf,N,m,mean_time=50,save=False,dist="Quantile Lorentzian",Folder_name="Review",Add_name="",K_draw=(1,13),r_draw=(0,0.9),Draw_theoretical = True):
    """
    To see total time vs order parameter, each time vs order parameter, coupling constant vs mean order parameter.
    At right graph, you can see the error bar, this mean standard deviation of order parameter.
    
    Args:
        Ksdf (Pandas DataFrame): After you done Hysteresis you can get Ksdf, Ksrdf
        Ksrdf (Pandas DataFrame): After you done Hysteresis you can get Ksdf, Ksrdf
        N (int): Number of oscillator that you put in Hysteresis argws.
        m (float): mass of oscillator that you put in Hysteresis argws.
        mean_time (int, optional): At right graph(`Coupling constant` vs `Order parameter`) you can control mean time. Defaults to 50.
        save (bool, optional): If you want to save file switch this to True. Defaults to False.
        dist (str, optional): You can change distribution of oscillator's natural frequency. So it will be change the theoretical Kc(critical coupling constant). Defaults to "Quantile Lorentzian". optional `"Lorentzian"`,`"Quantile Lorentzian"`, `"Nomal",`"Quantile Normal"`
        Folder_name (str, optional): Folder name where you want to save. Defaults to "Review".
        K_draw (tuple, optional): K xlim. Defaults to (1,13).
        r_draw (tuple, optional): r ylim. Defaults to (0,0.9).
    """
    Ks = Ksdf.index
    Ksr = Ksrdf.index
    int_ =np.linspace(0.0,1,len(Ks))
    color = plt.cm.viridis_r(int_)
    fig = plt.figure(figsize=(18,5))
    ax11,ax12 = plt.subplot(231),plt.subplot(234)
    ax21,ax22 = plt.subplot(232),plt.subplot(235)
    ax31 = plt.subplot(133)
    t_ = Ksdf["ts"].iloc[0]
    dK = Ks[1]-Ks[0]
    t_end = Ksdf["ts"].iloc[0][-1]
    dt = Ksdf["ts"].iloc[0][1]-Ksdf["ts"].iloc[0][0]
    num = 0
    for t_temp, r_temp in zip(Ksdf["ts"],Ksdf["rs"]):
        ax11.plot(t_temp,r_temp,color = color[num],label=f"{Ks[num]:.02f}")
        ax11.vlines(t_temp[0],-0.01,1.1,alpha=0.5,linestyles='--',color=color[num])
        if num%10==0:
            text_ = ax11.text(t_temp[0],max(r_temp)+0.2,f"K = {Ks[num]:.01f}",fontsize=8)
            text_.set_bbox(dict(facecolor=color[num], alpha=0.5, edgecolor=color[num]))
        num +=1

    for t_temp, r_temp in zip(Ksrdf["ts"],Ksrdf["rs"]):
        num -=1
        ax12.vlines(t_temp[0],-0.01,1.1,alpha=0.5,linestyles='--',color=color[num])
        if num%10==0:
            text_ = ax12.text(t_temp[-1],max(r_temp)+0.2,f"K = {Ks[num]:.01f}",fontsize=8)
            text_.set_bbox(dict(facecolor=color[num], alpha=0.5, edgecolor=color[num]))
        ax12.plot(t_temp,r_temp,color = color[num])
    ax12.invert_xaxis()
    ax11.set_ylabel("Order parameter")
    ax12.set_ylabel("Order parameter")
    ax11.set_xlabel(r"time [s] ($0 \rightarrow \infty$)")
    ax12.set_xlabel("time [s] ($\infty \leftarrow 0$)")
    # ax11.legend(ncol=6,fontsize=3,loc='center left', bbox_to_anchor=(1, 0))
    # plt.colorbar()
    ax11.text(0,0.9,"Forward")
    ax12.text(t_end*len(Ks),0.9,"Backward")
    ax11.grid()
    ax12.grid()
    fig.tight_layout()
    num = 0
    for t_temp, r_temp in zip(Ksdf["ts"],Ksdf["rs"]):
        ax21.plot(t_,r_temp,color = color[num],label=f"K = {Ks[num]:.02f}")
        num +=1
    
    for t_temp, r_temp in zip(Ksrdf["ts"],Ksrdf["rs"]):
        num -=1
        ax22.plot(t_,r_temp,color = color[num],label=f"K = {Ks[num]:.02f}")
    ax22.set_xlabel("time [s]")
    num = 0
    sca = plt.scatter(t_[0]*np.ones_like(Ks),r_temp[0]*np.ones_like(Ks),c=Ks,cmap='viridis_r',s=0)
    text_1 = ax21.text(0,0.8,"Forward")
    text_1.set_bbox(dict(facecolor=color[num], alpha=0.5, edgecolor=color[num]))
    text_2 = ax22.text(0,0.8,"Backward")
    text_2.set_bbox(dict(facecolor=color[num], alpha=0.5, edgecolor=color[num]))
    ax21.grid()
    ax22.grid()
    fig.tight_layout()
    r_last = []
    r_last_std = []
    r_r_last = []
    r_r_last_std = []
    Mean_time = int(mean_time/dt)
    for t_temp, r_temp in zip(Ksdf["ts"],Ksdf["rs"]):
        r_time_temp = r_temp[-Mean_time:]
        r_time_std = np.std(r_time_temp)
        r_last.append(np.mean(r_time_temp))
        r_last_std.append(r_time_std)
        
    for t_temp, r_temp in zip(Ksrdf["ts"],Ksrdf["rs"]):
        r_r_time_temp = r_temp[-Mean_time:]
        r_r_time_std = np.std(r_r_time_temp)
        r_r_last.append(np.mean(r_r_time_temp))
        r_r_last_std.append(r_r_time_std)
    if Draw_theoretical:
        Kspace = np.linspace(0.01,K_draw[1],1000)
        Kfwd,Rfwd,Kbwd,Rbwd = Make_theoretical_KR(Kspace,m)
        plt.plot(Kfwd,Rfwd,'.',alpha=0.4,markersize=1,label="Case 1(Blue,FW)",color = 'Tab:blue')
        plt.plot(Kbwd,Rbwd,'.',alpha=0.4,markersize=1,label="Case 2(Orange,BW)",color = 'Tab:orange')
        
    ax31.errorbar(Ks,r_last,yerr=r_last_std,label="Forward",fmt='d',markersize=6,capsize=3,color = 'Tab:blue')
    ax31.errorbar(Ks[::-1],r_r_last,yerr=r_r_last_std,label="Backward",fmt='d',markersize=6,capsize=3,color = 'Tab:orange')
    ax31.legend()
    for ax_ in (ax11,ax12,ax21,ax22):
        ax_.set_ylim(-0.05,1.05)

    ax31.grid()
    ax31.set_ylim(*r_draw)
    ax31.set_xlim(*K_draw)
    fig.tight_layout()
    ax31.set_ylabel("r (order parameter)",fontsize=13)
    ax31.set_xlabel("K (coupling constant)",fontsize=13)
    ax31.set_title(f"K vs r, m = {m:.02f}, N = {N}, dt = {dt}, dk = {dK:.02f}",fontsize=15)
    divider3 = make_axes_locatable(ax31)
    cax3 = divider3.append_axes("right", size="5%", pad="2%")
    cb3 = fig.colorbar(sca, cax=cax3)
    cb3.set_label("K (coupling constant)")
    fig.tight_layout()
    if save:
        plt.savefig(f"{Folder_name}/N = {N}, m = {m:.02f}, Total {dist},dk = {dK:.02f},{t_end},dt={dt}"+Add_name+".png",transparent=True,dpi = 500) 

from TO_sim.gen_Distribution import Quantile_Lorentzian   
def draw_theta_omega(theta,seed,N=500):
    fig = plt.figure()
    _,omega,_ = Quantile_Lorentzian(N, 0, 1)
    sca = plt.scatter(theta,omega,c=omega,vmin=-4,vmax=4)
    plt.ylim(-4,4)
    ax = plt.gca()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar =fig.colorbar(sca, cax=cax, orientation='vertical')
    cbar.set_label(r'$\omega$(natural frequency)',fontsize=13)
    ax.set_xlabel(r'$\theta$[rad](phase)',fontsize=13)
    ax.set_ylabel(r'$\omega$[rad/s]'+'\n (natural frequency)',fontsize=13)
    ax.set_title(r'$\theta$ vs $\omega$'+r', $\theta$'+f' distribution = {seed}',fontsize=15)
    fig.tight_layout()
    return fig

def Draw_slicing_graph(df,m,reverse=False,save=True,Folder_name ='Review',dt = 0.1,Slice_time=15,dK=0.2):
    Ks = df.index
    slicing = lambda x,sec:x[-int(sec/dt):]
    sliced_r = df.rs.apply(slicing,sec=50)
    def make_marker(x):
        x = x.copy()
        x[0,:]=+20
        x[1,:]=-20
        x[2,:]=+20
        return x
    ST = Slice_time
    
    data = np.array([make_marker(df.dtheta_s.iloc[i][-ST*10:,:]) for i in range(len(Ks))])
    data_rs = [df.rs.iloc[i][-ST*10:] for i in range(len(Ks))]
    A = np.concatenate(data,axis=0)
    rs_total = np.concatenate(data_rs,axis=0)
    ts_total = np.arange(len(rs_total))*0.1
    ts_marker = np.arange(len(Ks))*ST
    Ks_marker = np.arange(len(Ks))*0.2 +0.1
    RS = np.split(rs_total,len(Ks))
    TS = np.split(ts_total,len(Ks))
    fig = plt.figure(figsize=(15,5),dpi=200)
    int_ =np.linspace(0.0,1,len(Ks))
    if reverse: 
        int_ = int_[::-1]
    color = plt.cm.viridis(int_)
    plt.subplot(211)
    im11 = plt.imshow(A.T,origin='lower',extent=[Ks[0],Ks[-1],0,500],vmin=-3,vmax=3,aspect='auto')
    plt.xlabel('K : coupling constant',fontsize=13)
    plt.ylabel('$i$-th oscillator',fontsize=13)
    # plt.vlines(Ks_marker,[0],[500],ls=':',color='red',alpha=0.3)
    plt.hlines(250,[Ks[0]],[Ks[-1]],ls=':',color='black',alpha=0.3)
    if reverse:
        ax = plt.gca()
        ax.invert_xaxis()


    ################# additional axes #############
    ax11 = plt.gca()
    divider11 = make_axes_locatable(ax11)
    cax11 = divider11.append_axes("right", size="1%", pad=0.05)
    colorbar2 = fig.colorbar(im11, cax=cax11, orientation="vertical", extend="both")
    colorbar2.set_label(r'$\dot{\theta}$: phase vel.')

    ############### New Graph ###################
    plt.subplot(212)
    # plt.plot(ts_total,rs_total)
    sca = plt.scatter(-1*np.ones(len(Ks)),-1*np.ones(len(Ks)),c = Ks,s=0)
    for i,(t,r) in enumerate(zip(TS,RS)):
        plt.plot(t,r,color=color[i])
    plt.ylim(0,0.9)
    plt.xlim(0,ts_total[-1])
    plt.ylabel('r : order parameter',fontsize=15)
    if reverse:
        ax = plt.gca()
        ax.invert_xaxis()
        plt.xlabel("time [s] ($\infty \leftarrow 0$)",fontsize=15)
        
    else:
        plt.xlabel(r"time [s] ($0 \rightarrow \infty$)",fontsize=15)
    plt.vlines(ts_marker,[0],[1],ls=':',color='red',alpha=0.3,label=f'slicing last {ST}s each K')
    plt.legend()
    ################# additional axes #############
    ax2 = plt.gca()
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="1%", pad=0.05)
    # cax2.axis('off')
    colorbar2 =fig.colorbar(sca, cax=cax2, orientation="vertical")
    colorbar2.set_label('K')

    plt.suptitle(f'm = {m}, dK = {dK}, N = 500',y=0.95,fontsize=18)
    plt.tight_layout()
    # plt.savefig(DF+f'm ={m} t vs r + K vs phase vel reverse.png',dpi=400)
    if save:
        if reverse:
            plt.savefig(Folder_name+f'm ={m} t vs r + K vs phase vel backward.png',dpi=400,transparent=True)
        else:
            plt.savefig(Folder_name+f'm ={m} t vs r + K vs phase vel foward.png',dpi=400,transparent=True)
    plt.show()
    
            
from TO_sim.Check_theoretical import *
class Draw_theoretical():
    def __init__(self,m):
        self.m = m
        Ks = np.linspace(1,13,1000)
        KF,RF,KB,RB = map(np.array,Make_theoretical_KR(Ks,m))
        self.KF =KF
        self.KB =KB
        self.RF =RF
        self.RB =RB
        First = RB[0]
        idx = np.where(RB>First)
        notidx = np.where(RB<First)
        self.RBU = RB[idx]
        self.KBU = KB[idx]
        self.RBD = RB[notidx]
        self.KBD = KB[notidx]
    def backward(self,label = True):
        if self.m==0:
            if label:
                bw, = plt.plot(self.KB,self.RB,color = 'Tab:Orange',label='Backward theoretical')
            else:
                bw, = plt.plot(self.KB,self.RB,color = 'Tab:Orange') 
            return bw
        else:
            if label:
                bw1, = plt.plot(self.KBU,self.RBU,color = 'Tab:Orange',label='Backward theoretical')
                bw2, = plt.plot(self.KBD,self.RBD,color = 'Tab:Orange')
            else:
                bw1, = plt.plot(self.KBU,self.RBU,color = 'Tab:Orange')
                bw2, = plt.plot(self.KBD,self.RBD,color = 'Tab:Orange')
            return bw1,bw2

    def foward(self,label = True):
        if label:
            fw, = plt.plot(self.KF,self.RF,color = 'Tab:blue',label='Forward theoretical')
        else:
            fw, =plt.plot(self.KF,self.RF,color = 'Tab:blue')
        return fw

    def total(self,label=True):
        if label:    
            if self.m==0:
                fw, = plt.plot(self.KF,self.RF,color = 'Tab:blue',label='Forward theoretical')
                bw, = plt.plot(self.KB,self.RB,color = 'Tab:Orange',label='Backward theoretical')
                return fw,bw
            else:
                fw, =plt.plot(self.KF,self.RF,color = 'Tab:blue',label='Forward theoretical')
                bw1, = plt.plot(self.KBU,self.RBU,color = 'Tab:Orange',label='Backward theoretical')
                bw2, = plt.plot(self.KBD,self.RBD,color = 'Tab:Orange')
                return fw,(bw1,bw2)
        else:
            if self.m==0:
                fw, = plt.plot(self.KF,self.RF,color = 'Tab:blue')
                bw, = plt.plot(self.KB,self.RB,color = 'Tab:Orange')
                return fw,bw
            else:
                fw, = plt.plot(self.KF,self.RF,color = 'Tab:blue')
                bw1, = plt.plot(self.KBU,self.RBU,color = 'Tab:Orange')
                bw2, = plt.plot(self.KBD,self.RBD,color = 'Tab:Orange')
                return fw,(bw1,bw2)
from TO_sim.analytical.order_parameter import  Make_empirical_KR
class Draw_theoretical_wData():
    def __init__(self,m,dist="normal"):
        self.m = m
        KF,RF,KB,RB = Make_empirical_KR(m,dist=dist)
        self.KF =KF
        self.KB =KB
        self.RF =RF
        self.RB =RB
        First = RB[0]
        idx = np.where(RB>First)
        notidx = np.where(RB<First)
        self.RBU = RB[idx]
        self.KBU = KB[idx]
        self.RBD = RB[notidx]
        self.KBD = KB[notidx]
    def backward(self,label = True):
        if self.m==0:
            if label:
                bw, = plt.plot(self.KB,self.RB,color = 'Tab:Orange',label='Backward theoretical')
            else:
                bw, = plt.plot(self.KB,self.RB,color = 'Tab:Orange') 
            return bw
        else:
            if label:
                bw1, = plt.plot(self.KBU,self.RBU,color = 'Tab:Orange',label='Backward theoretical')
                bw2, = plt.plot(self.KBD,self.RBD,color = 'Tab:Orange')
            else:
                bw1, = plt.plot(self.KBU,self.RBU,color = 'Tab:Orange')
                bw2, = plt.plot(self.KBD,self.RBD,color = 'Tab:Orange')
            return bw1,bw2

    def foward(self,label = True):
        if label:
            fw, = plt.plot(self.KF,self.RF,color = 'Tab:blue',label='Forward theoretical')
        else:
            fw, =plt.plot(self.KF,self.RF,color = 'Tab:blue')
        return fw

    def total(self,label=True):
        if label:    
            if self.m==0:
                fw, = plt.plot(self.KF,self.RF,color = 'Tab:blue',label='Forward theoretical')
                bw, = plt.plot(self.KB,self.RB,color = 'Tab:Orange',label='Backward theoretical')
                return fw,bw
            else:
                fw, =plt.plot(self.KF,self.RF,color = 'Tab:blue',label='Forward theoretical')
                bw1, = plt.plot(self.KBU,self.RBU,color = 'Tab:Orange',label='Backward theoretical')
                bw2, = plt.plot(self.KBD,self.RBD,color = 'Tab:Orange')
                return fw,(bw1,bw2)
        else:
            if self.m==0:
                fw, = plt.plot(self.KF,self.RF,color = 'Tab:blue')
                bw, = plt.plot(self.KB,self.RB,color = 'Tab:Orange')
                return fw,bw
            else:
                fw, = plt.plot(self.KF,self.RF,color = 'Tab:blue')
                bw1, = plt.plot(self.KBU,self.RBU,color = 'Tab:Orange')
                bw2, = plt.plot(self.KBD,self.RBD,color = 'Tab:Orange')
                return fw,(bw1,bw2)
                
def Draw_mean_graph(df,m,Folder_name ='Review',reverse=False,save=True,dK=0.2,Slicing_time=50):
    ST = Slicing_time
    Ks = df.index
    Draw_ = Draw_theoretical(m)
    data = np.array([(np.mean(df.dtheta_s.iloc[i][-ST*10:,:],axis=0)) for i in range(len(Ks))])
    data_rs = [np.mean(df.rs.iloc[i][-ST*10:]) for i in range(len(Ks))]
    data_std = [np.std(df.rs.iloc[i][-ST*10:]) for i in range(len(Ks))]
    fig = plt.figure(figsize=(15,5),dpi=300)
    int_ =np.linspace(0.0,1,len(Ks))
    if reverse: 
        int_ = int_[::-1]
    color = plt.cm.viridis(int_)
    plt.subplot(211)
    im11 = plt.imshow(data.T,origin='lower',extent=[Ks[0],Ks[-1],0,500],vmin=-3,vmax=3,aspect='auto',cmap='viridis')
    plt.xlabel('K : coupling constant',fontsize=13)
    plt.ylabel('$i$-th oscillator',fontsize=13)
    plt.hlines(250,[Ks[0]],[Ks[-1]],ls=':',color='black',alpha=0.3)
    # plt.grid()
    if reverse:
        ax = plt.gca()
        ax.invert_xaxis()
    ################# additional axes #############
    ax11 = plt.gca()
    divider11 = make_axes_locatable(ax11)
    cax11 = divider11.append_axes("right", size="1%", pad=0.05)
    colorbar2 = fig.colorbar(im11, cax=cax11, orientation="vertical", extend="both")
    colorbar2.set_label(r'$\dot{\theta}$: phase vel.')

    ############### K vs r ###################
    plt.subplot(212)
    
    
    plt.ylim(0,0.9)
    plt.xlim(Ks[0],Ks[-1])
    plt.ylabel('r : order parameter',fontsize=15)
    plt.grid()
    plt.xlabel('K : coupling constant',fontsize=13)
    
    if reverse:
        # sca = plt.scatter(Ks,data_rs,c=Ks,marker='.')
        plt.errorbar(Ks,data_rs,data_std,fmt='.',alpha=0.3,color='tab:orange',capsize=2,label='Case 2(Orange,BW)')
        ax = plt.gca()
        Draw_.backward()
        ax.invert_xaxis()
    else:
        # sca = plt.scatter(Ks,data_rs,c=Ks,marker='.')
        plt.errorbar(Ks,data_rs,data_std,fmt='.',alpha=0.3,color='tab:blue',capsize=2,label='Case 1(Blue,FW)')
        Draw_.foward()
    plt.legend()
    ################# additional axes #############
    ax2 = plt.gca()
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="1%", pad=0.05)
    cax2.axis('off')
    # colorbar2 =fig.colorbar(sca, cax=cax2, orientation="vertical")
    colorbar2.set_label('K')

    plt.suptitle(f'm = {m}, dK = {dK}, N = 500',y=0.95,fontsize=18)
    plt.tight_layout()
    # plt.savefig(DF+f'm ={m} t vs r + K vs phase vel reverse.png',dpi=400)
    if save:
        if reverse:
            plt.savefig(Folder_name+f'm ={m}, dK={dK} K vs r + K vs phase vel backward.png',dpi=400,transparent=True)
        else:
            plt.savefig(Folder_name+f'm ={m}, dK={dK} K vs r + K vs phase vel foward.png',dpi=400,transparent=True)
    plt.show()
    
            
def Draw_simple_Kr(df,rdf,m,Folder_name ='Review',save=True,dK=0.2,Slicing_time=50,label = True,alpha = 1):
    ST = Slicing_time
    Ks = df.index
    Ksr = rdf.index
    data_rs = [np.mean(df.rs.iloc[i][-ST*10:]) for i in range(len(Ks))]
    data_rrs = [np.mean(rdf.rs.iloc[i][-ST*10:]) for i in range(len(Ks))]

    Draw_ = Draw_theoretical(m)
    Draw_.total(label)
    if label:
        plt.plot(Ks,data_rs,'.',label=r"$Forward$",markersize=6,color = 'Tab:Blue',alpha=alpha)
        plt.plot(Ksr,data_rrs,'.',label=r"$Backward$",markersize=6,color = 'Tab:Orange',alpha=alpha)
        plt.legend()  
    else: 
        plt.plot(Ks,data_rs,'.',markersize=6,color = 'Tab:Blue',alpha=alpha)
        plt.plot(Ksr,data_rrs,'.',markersize=6,color = 'Tab:Orange',alpha=alpha)
    plt.title(f'K vs r graph, m = {m}',fontsize= 15)
    plt.grid()
      
    plt.xlim(1,13)
    plt.ylim(0,0.9)

    plt.xlabel('K : Coupling constant',fontsize=13)
    plt.ylabel('r : Order parameter',fontsize=13)
    plt.tight_layout()
    if save:
        plt.savefig(Folder_name+f'simple ver. Hystersis m={m},dK = {dK}.png',dpi=400,transparent=True)
        
        
### VER2 ####
def Draw_avg_vel_r(t_sum,avg_r,avg_dtheta,KmN,figsize=(6.4,2)):
    """_summary_
    put time, average r, average phase velocity, K,m,N, figsize
    return Draw time vs avg_r and Oscillator number vs avg. phase. velocity
    Args:
        t_sum (_type_): t[sum_time:]
        avg_r (_type_): average order parameter
        avg_dtheta (): average phase velocity
        KmN (tuple): (K,m,N)
        figsize (tuple, optional): figure size. Defaults to (6.4,2).
    """
    K,m,N = KmN
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=figsize)
    plt.sca(ax1),plt.plot((avg_dtheta[-1]))
    plt.xlabel('oscillator index'),plt.ylabel('avg ang. vel.')
    plt.title(r'oscillator idx. vs avg. $\dot{\theta}$')
    
    plt.sca(ax2),plt.plot(t_sum,avg_r),plt.ylim(0,1)
    plt.xlabel('time'),plt.ylabel('avg $r$')
    plt.title(r'time vs avg. $r$')
    plt.suptitle(f'$K$ = {K}, $m$ = {m}, $N$ = {N}',y=0.85)
    plt.tight_layout()