from IPython.display import HTML
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable



def Animation_logabs(Ksdf, Ksrdf, N, m,MaxFrame=10000):
    fig = plt.figure(facecolor="white")
    ax11 = plt.subplot(221)
    ax12 = plt.subplot(222)
    ax22 = plt.subplot(212)
    logabs = lambda x: np.log(np.abs(x))
    def Slicing(x):
        slice_ = (Ksdf["dtheta_s"].iloc[0].shape[0] //MaxFrame)
        slice_ = 1 if slice_<=1 else slice_
        return x[::slice_,:]
    Temp_Ks = Ksdf["dtheta_s"].apply(logabs).apply(Slicing)
    Temp_Ksr = Ksrdf["dtheta_s"].apply(logabs).apply(Slicing)
    Ks = Ksdf.index
    Temp_rs = Ksdf["rs"]
    Temp_rsr = Ksrdf["rs"]
    Temp_t = Ksdf["ts"].iloc[0]
    t_end = Ksdf["ts"].iloc[0][-1]
    Ks = Ksdf.index
    K = Ks[0]

    ax11.clear()
    ax12.clear()
    ax11.set_xlabel(f"Oscillator No.(N={N})", fontsize=12)
    ax12.set_xlabel(f"Oscillator No.(N={N})", fontsize=12)
    ax11.set_title(f"Forward", fontsize=12)
    ax12.set_title(f"Backward", fontsize=12)
    ax12.tick_params(labelleft = False )
    ax11.set_ylabel("phase velocity\n" + r"($\log{|\dot{\theta}|}$)", fontsize=12)
    im11 = ax11.imshow(
        Temp_Ks[K], extent=[0, N, t_end, 0], aspect="auto", vmin=-8, vmax=3.4
    )
    im12 = ax12.imshow(
        Temp_Ksr[K], extent=[0, N, t_end, 0], aspect="auto", vmin=-8, vmax=3.4
    )
    # ax11.text(0,-5,f"m = {m}, K = {K}",fontsize=15)

    ax22.clear()
    ax22.plot(Temp_t, Temp_rs[K], label="Forward")
    ax22.plot(Temp_t, Temp_rsr[K], label="Backward")
    ax22.set_xlabel("Time[s]", fontsize=12)
    ax22.set_ylabel("Order\nparameter(r)", fontsize=12)
    ax22.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.20),
        fancybox=True,
        shadow=True,
        ncol=2,
    )
    ax22.set_ylim(0, 1)
    ax22.grid()
    divider11 = make_axes_locatable(ax11)
    divider12 = make_axes_locatable(ax12)
    cax11 = divider11.append_axes("right", size="5%", pad=0.05)
    cax12 = divider12.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im11, cax=cax11, orientation="vertical", extend="both")
    fig.colorbar(im12, cax=cax12, orientation="vertical", extend="both")
    fig.suptitle(f"m = {m}, K = {K}", fontsize=15, position=(0.5, 0.92))

    # fig.colorbar(pcm, ax = [ax11,ax12],location='bottom')

    # ax2.text(0,-5,f"K = {K}",fontsize=15)
    fig.tight_layout()

    def Update(K):
        ax11.clear()
        ax12.clear()
        ax11.set_title(f"Forward", fontsize=12)
        ax12.set_title(f"Backward", fontsize=12)
        ax11.set_xlabel(f"Oscillator No.(N={N})", fontsize=12)
        ax12.set_xlabel(f"Oscillator No.(N={N})", fontsize=12)
        ax11.set_ylabel("phase velocity\n" + r"($\log{|\dot{\theta}|}$)", fontsize=12)

        ax11.imshow(
            Temp_Ks[K], extent=[0, N, t_end, 0], aspect="auto", vmin=-8, vmax=3.4
        )
        ax12.imshow(
            Temp_Ksr[K], extent=[0, N, t_end, 0], aspect="auto", vmin=-8, vmax=3.4
        )
        # ax11.text(0,-5,f"m = {m}, K = {K}",fontsize=15)

        ax22.clear()
        ax22.plot(Temp_t, Temp_rs[K], label="Forward")
        ax22.plot(Temp_t, Temp_rsr[K], label="Backward")
        ax22.set_xlabel("Time[s]", fontsize=12)
        ax22.set_ylabel("Order\nparameter (r)", fontsize=12)
        ax22.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.20),
            fancybox=True,
            shadow=True,
            ncol=2,
        )
        ax22.set_ylim(0, 1)
        ax22.grid()
        fig.suptitle(f"m = {m}, K = {K}", fontsize=15)
        # fig.colorbar(im11, cax=cax11, orientation='vertical')
        # fig.colorbar(im12, cax=cax12, orientation='vertical')
        # ax2.text(0,-5,f"K = {K}",fontsize=15)
        fig.tight_layout()

    ani = FuncAnimation(fig, Update, frames=Ks, interval=500)
    return ani

