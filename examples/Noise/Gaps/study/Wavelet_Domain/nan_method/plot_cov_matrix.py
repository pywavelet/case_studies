import matplotlib.pyplot as plt 
import numpy as np

Cov_Matrix = np.load("matrix_directory/Cov_Matrix_Flat_w_filter.npy")

# title_label = [r'Analytical $|\Sigma_{N}(f,f^{\prime})|$']

fig, ax = plt.subplots(figsize = (14,10))
mat_gap_mine = ax.matshow(np.log10(np.abs(Cov_Matrix)))#, vmin = -80, vmax = -50)
cbar = fig.colorbar(mat_gap_mine, ax=ax, location='right', 
             shrink=0.8)
# cbar.set_label(fontsize = 10)
cbar.ax.tick_params(labelsize=10) 


plt.tight_layout() 
plt.show()