import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

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

def Time_R_df_total(Ksdf,Ksrdf,N,m,mean_time=50,save=False,dist="Quantile Lorentzian",Folder_name="Review",K_draw=(1,13),r_draw=(0,0.9)):
    """_summary_
    To see total time vs order parameter, each time vs order parameter, coupling constant vs mean order parameter.
    At right graph, you can see the error bar, this mean standard deviation of order parameter.
    
    Args:
        Ksdf (Pandas DataFrame): After you done Hysteresis you can get Ksdf, Ksrdf
        Ksrdf (Pandas DataFrame): After you done Hysteresis you can get Ksdf, Ksrdf
        N (int): Number of oscillator that you put in Hysteresis argws.
        m (float): mass of oscillator that you put in Hysteresis argws.
        mean_time (int, optional): At right graph(`Coupling constant` vs `Order parameter`) you can control mean time. Defaults to 50.
        save (bool, optional): If you want to save file switch this to True. Defaults to False.
        dist (str, optional): You can change distribution of oscillator's natural frequency. So it will be change the theorical Kc(critical coupling constant). Defaults to "Quantile Lorentzian". optional `"Lorentzian"`,`"Quantile Lorentzian"`, `"Nomal",`"Quantile Normal"`
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
        
    ax31.errorbar(Ks,r_last,yerr=r_last_std,label="Forward",fmt='d',markersize=6,capsize=3)
    ax31.errorbar(Ks[::-1],r_r_last,yerr=r_r_last_std,label="Backward",fmt='d',markersize=6,capsize=3)
    ax31.legend()
    for ax_ in (ax11,ax12,ax21,ax22):
        ax_.set_ylim(-0.05,1.05)

    ax31.grid()
    ax31.set_ylim(*r_draw)
    ax31.set_xlim(*K_draw)
    fig.tight_layout()
    ax31.set_ylabel("r (order parameter)",fontsize=13)
    ax31.set_xlabel("K (coupling constant)",fontsize=13)
    ax31.set_title(f"K vs r, m = {m}, N = {N}, dt = {dt}",fontsize=15)
    divider3 = make_axes_locatable(ax31)
    cax3 = divider3.append_axes("right", size="5%", pad="2%")
    cb3 = fig.colorbar(sca, cax=cax3)
    cb3.set_label("K (coupling constant)")
    fig.tight_layout()
    if save:
        plt.savefig(f"{Folder_name}/N = {N}, m = {m}, Total {dist},dk = {dK:.02f},{t_end},dt={dt}.png",transparent=True,dpi = 500)     