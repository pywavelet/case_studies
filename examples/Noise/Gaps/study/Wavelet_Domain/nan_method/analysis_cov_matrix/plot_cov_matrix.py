import matplotlib.pyplot as plt
import numpy as np


def plot_cov_matrix(file_path, output_path):
    cov_matrix = np.load(file_path)
    fig, ax = plt.subplots(figsize=(14, 10))
    mat_gap_mine = ax.matshow(np.log10(np.abs(cov_matrix)))
    cbar = fig.colorbar(mat_gap_mine, ax=ax, location="right", shrink=0.8)
    cbar.ax.tick_params(labelsize=10)
    plt.tight_layout()
    plt.savefig(output_path)


if __name__ == "__main__":
    plot_cov_matrix(
        "matrix_directory/Cov_Matrix_Flat_w_filter.npy",
        "matrix_directory/Cov_Matrix_Flat_w_filter.png",
    )
    plot_cov_matrix(
        "matrix_directory/Cov_Matrix_Flat_w_filter_gap.npy",
        "matrix_directory/Cov_Matrix_Flat_w_filter_gap.png",
    )
