import numpy as np
import matplotlib.pyplot as plt

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

        