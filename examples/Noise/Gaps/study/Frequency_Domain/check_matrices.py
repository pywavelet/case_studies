import os

import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "Data")


def main():
    print("Now loading in analytical cov matrix")
    Cov_Matrix_analytical_gap_TDI1 = np.load(
        f"{DATA_DIR}/Cov_Matrix_analytical_gap_TDI1.npy"
    )
    Cov_Matrix_analytical_gap_TDI2 = np.load(
        f"{DATA_DIR}/Cov_Matrix_analytical_gap_TDI2.npy"
    )
    Cov_Matrix_analytical_gap_Cornish = np.load(
        f"{DATA_DIR}/Cov_Matrix_analytical_gap_cornish.npy"
    )
    print("Now loading in estimated covariance matrix")
    Cov_Matrix_estm_gap = np.load(f"{DATA_DIR}/Cov_Matrix_estm_gap.npy")

    # fig,ax = plt.subplots(2,2, figsize = (16,8))
    # j = 0

    # ax[0,0].semilogy(np.abs(np.diag(Cov_Matrix_analytical_gap,j)),'*',label = 'analytical')
    # ax[0,0].semilogy(np.abs(np.diag(Cov_Matrix_estm_gap,j)),alpha = 0.7,label = 'estimated')
    # ax[0,0].set_title('Comparison of covariance matrices: Diagonal = {0}'.format(j))

    # ax[0,1].semilogy(np.abs(np.diag(Cov_Matrix_analytical_gap,j+10)),'*',label = 'analytical')
    # ax[0,1].semilogy(np.abs(np.diag(Cov_Matrix_estm_gap,j+10)),alpha = 0.7,label = 'estimated')
    # ax[0,1].set_title('Comparison of covariance matrices: Diagonal = {0}'.format(j+10))

    # ax[1,0].semilogy(np.abs(np.diag(Cov_Matrix_analytical_gap,j+50)),'*',label = 'analytical')
    # ax[1,0].semilogy(np.abs(np.diag(Cov_Matrix_estm_gap,j+50)),alpha = 0.7,label = 'estimated')
    # ax[1,0].set_title('Comparison of covariance matrices: Diagonal = {0}'.format(j+50))

    # ax[1,1].semilogy(np.abs(np.diag(Cov_Matrix_analytical_gap,j+100)),'*',label = 'analytical')
    # ax[1,1].semilogy(np.abs(np.diag(Cov_Matrix_estm_gap,j+100)),alpha = 0.7,label = 'estimated')
    # ax[1,1].set_title('Comparison of covariance matrices: Diagonal = {0}'.format(j+100))

    # for i in range(0,2):
    #     for j in range(0,2):
    #         ax[i,j].set_xlabel(r'index of matrix')
    #         ax[i,j].set_ylabel(r'magnitude')
    #         ax[i,j].legend()
    #         ax[i,j].grid()
    # plt.tight_layout()
    # plt.show()

    # Analyse the two inverse matrices together -- see that they are dense as fuck

    N_time = 2 * (Cov_Matrix_analytical_gap_TDI1.shape[0] - 1)

    freq_bin = np.fft.rfftfreq(N_time, 66)
    title_label = [r"TDI1", r"TDI2", "Cornish"]

    fig, ax = plt.subplots(1, 3, figsize=(16, 8))

    mat_TDI1 = ax[0].matshow(
        np.log10(np.abs(Cov_Matrix_analytical_gap_TDI1)), vmin=-47, vmax=-36
    )
    mat_TDI2 = ax[1].matshow(
        np.log10(np.abs(Cov_Matrix_analytical_gap_TDI2)), vmin=-47, vmax=-36
    )
    mat_cornish = ax[2].matshow(
        np.log10(np.abs(Cov_Matrix_analytical_gap_Cornish)), vmin=-47, vmax=-36
    )

    cbar = fig.colorbar(
        mat_cornish, ax=ax, location="top", anchor=(0, -16), shrink=1
    )
    cbar.set_label(label=r"Power", fontsize=30)
    cbar.ax.tick_params(labelsize=20)

    ticks = [0, 100, 200, 300, 400, 512]
    freq_ticks = [freq_bin[item] for item in ticks]

    ax[1].yaxis.tick_right()

    for i in range(0, 3):
        ax[i].set_xticks(
            ticks, np.round(freq_ticks, 3), fontsize=12, rotation=315
        )
        ax[0].set_yticks(
            ticks, np.round(freq_ticks, 3), fontsize=12, rotation=45
        )
        ax[1].set_yticks(
            ticks, np.round(freq_ticks, 3), fontsize=12, rotation=315
        )
        ax[i].set_title("Frequency [Hz]", fontsize=20)
        ax[0].set_ylabel("Frequency [Hz]", fontsize=20)
        ax[i].set_xlabel(title_label[i], fontsize=22)
    plt.tight_layout()
    plt.savefig(f"{DATA_DIR}/Covariance_matrix_comparison.png")
    print("Saved matrix comparison plot")


if __name__ == "__main__":
    main()
