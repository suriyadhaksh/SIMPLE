import numpy as np
import matplotlib.pyplot as plt
import matplotlib 

matplotlib.rcParams.update({'font.size': 10, 'font.family': 'sans-serif'})
matplotlib.rcParams.update({'text.usetex': 'true'})

file_vec = ['Re_100.npz', 'Re_400.npz', 'Re_1000.npz']
Re_vec = [100, 400, 1000]

# Ghia et al. (1982) reference solution
y_midsection_vec    = [1.00000, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344, 0.6172,
        0.5000, 0.4531, 0.2813, 0.1719, 0.1016, 0.0703, 0.0625, 0.0547, 0.0000]

u100  = [ 1.00000,  0.84123,  0.78871,  0.73722,  0.68717,  0.23151,  0.00332, -0.13641,
         -0.20581, -0.21090, -0.15662, -0.10150, -0.06434, -0.04775, -0.04192, -0.03717,  0.00000]

u400  = [ 1.00000,  0.75837,  0.68439,  0.61756,  0.55892,  0.29093,  0.16256,  0.02135,
         -0.11477, -0.17119, -0.32726, -0.24299, -0.14612, -0.10338, -0.09266, -0.08186,  0.00000]

u1000 = [ 1.00000,  0.65928,  0.57492,  0.51117,  0.46604,  0.33304,  0.18719,  0.05702,
         -0.06080, -0.10648, -0.27805, -0.38289, -0.29730, -0.22220, -0.20196, -0.18109,  0.00000]


x_midsection_vec = [
    1.0000, 0.9688, 0.9609, 0.9531, 0.9453, 0.9063, 0.8594, 0.8047,
    0.5000, 0.2344, 0.2266, 0.1563, 0.0938, 0.0781, 0.0703, 0.0625,
    0.0000
]

v100 = [
     0.00000, -0.05906, -0.07391, -0.08864, -0.10313, -0.16914, -0.22445, -0.24533,
     0.05454,  0.17527,  0.17507,  0.16077,  0.12317,  0.10890,  0.10091,  0.09233,
     0.00000
]

v400 = [
     0.00000, -0.12146, -0.15663, -0.19254, -0.22847, -0.23827, -0.44993, -0.38598,
     0.05186,  0.30174,  0.30203,  0.28124,  0.22965,  0.20920,  0.19713,  0.18360,
     0.00000
]

v1000 = [
     0.00000, -0.21388, -0.27669, -0.33714, -0.39188, -0.51550, -0.42665, -0.31966,
     0.02526,  0.32235,  0.33075,  0.37095,  0.32627,  0.30353,  0.29012,  0.27485,
     0.00000
]

u_ghia_sol = [u100, u400, u1000]
v_ghia_sol = [v100, v400, v1000]

fig, axs = plt.subplots(1, 2, figsize=(10, 5))

axs[0].set_xlabel('$u$')
axs[0].set_ylabel('$y$')
#axs[0].set_title('At $x=0.5$')

axs[1].set_xlabel('$x$')
axs[1].set_ylabel('$v$')
#axs[1].set_title('At $y=0.5$')

color_vec = ['r', 'g', 'b']

for i in range(len(file_vec)):
    # Load data
    data = np.load(file_vec[i])
    if 'x' in data and 'y' in data:
        x1d, y1d = data['x'], data['y']
        X, Y = np.meshgrid(x1d, y1d)
    elif 'X' in data and 'Y' in data:
        X, Y = data['X'], data['Y']
    else:
        raise KeyError("Dataset must contain either ('x','y') or ('X','Y') keys")
    u, v, p = data['u'], data['v'], data['p']
    

    # Interpolated line-cuts at x=0.5 and y=0.5
    x_cut = 0.5
    y_cut = 0.5

    # _-- u vs y at x=0.5 ---
    y_line    = y1d
    u_xslice  = np.array([np.interp(x_cut, x1d, u[j, :]) for j in range(len(y1d))])
    print(f"Interpolated at x={x_cut} for all y → {len(y_line)} points")
    axs[0].plot(u_xslice, y_line,    color_vec[i]+'--', label=f'{Re_vec[i]}')
    axs[0].plot(u_ghia_sol[i], y_midsection_vec,
                color_vec[i]+'o', markerfacecolor='white',
                markersize=5,  label=f'{Re_vec[i]} (Ghia)')

    # --- v vs x at y=0.5 ---
    x_line    = x1d
    v_yslice  = np.array([np.interp(y_cut, y1d, v[:, j]) for j in range(len(x1d))])
    print(f"Interpolated at y={y_cut} for all x → {len(x_line)} points")
    axs[1].plot(x_line,    v_yslice,    color_vec[i]+'--', label=f'{Re_vec[i]}')
    axs[1].plot(x_midsection_vec, v_ghia_sol[i], 
                color_vec[i]+'o',  markerfacecolor='white',
                markersize=5,   label=f'{Re_vec[i]} (Ghia)')


axs[0].legend()
axs[1].legend()   
axs[0].set_ylim(0, 1)
axs[1].set_xlim(0, 1)


#fig.patch.set_alpha(0.0)
plt.tight_layout()  
plt.savefig(f'midsection.png',transparent=False)
plt.close(fig)

    

