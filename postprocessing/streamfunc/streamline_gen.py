import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({'font.size': 10, 'font.family': 'sans-serif'})
matplotlib.rcParams.update({'text.usetex': 'true'})

file_vec = ['Re_100.npz', 'Re_400.npz', 'Re_1000.npz']
Re_vec   = [100, 400, 1000]

# Ghia et al. (1982) reference streamfunction levels
ghia_streamfunction_levels = [-1e-10, -1e-7, -1e-5, -1e-4, -0.01, -0.03, -0.05, -0.07, -0.09, -0.1, -0.11, -0.115, -0.1175, 1e-6, 1e-5, 5e-5]
ghia_streamfunction_labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', '2', '3', '4']

# Zip and sort levels and labels together
sorted_pairs = sorted(zip(ghia_streamfunction_levels, ghia_streamfunction_labels), key=lambda x: x[0])
levels, labels = zip(*sorted_pairs)

for i, fname in enumerate(file_vec):
    
    # Load data
    data = np.load(fname)
    if 'x' in data and 'y' in data:
        x1d, y1d = data['x'], data['y']
        X, Y = np.meshgrid(x1d, y1d)
    elif 'X' in data and 'Y' in data:
        X, Y = data['X'], data['Y']
    else:
        raise KeyError("Dataset must contain either ('x','y') or ('X','Y') keys")
    u, v, p = data['u'], data['v'], data['p']

    dx = x1d[1] - x1d[0]
    dy = y1d[1] - y1d[0]

    # Compute streamfunction ψ by integrating u in y-direction
    psi_grid = np.zeros_like(u)
    for j in range(1, u.shape[0]):
        psi_grid[j, :] = psi_grid[j-1, :] + u[j, :] * dy

    # psi_grid = np.rot90(np.fliplr(psi_grid),k=1)
    

    # Plot
    fig, ax = plt.subplots()

    cs = ax.contour(X, Y, psi_grid, levels=levels, colors='k', linewidths=0.7)

    # Label contours
    fmt = {lvl: lbl for lvl, lbl in zip(levels, labels)}
    ax.clabel(cs, inline=True, fontsize=13, fmt=fmt)

    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal') 

    plt.tight_layout(pad=0)
    fig.savefig(f'Re_{Re_vec[i]}.png', transparent=False, bbox_inches='tight')
    plt.close(fig)
