import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({'font.size': 10, 'font.family': 'sans-serif'})
matplotlib.rcParams.update({'text.usetex': 'true'})

file_vec = ['Re_100.npz', 'Re_500.npz', 'Re_1000.npz']
Re_vec   = [100, 500, 1000]
n_stream = 20
n_levels = 50

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

    fig, ax = plt.subplots()

    # Contour of p
    levels = np.linspace(p.min(), p.max(), n_levels)
    cf     = ax.contourf(X, Y, p, levels=levels, cmap='viridis', alpha=0.8)
    cbar   = fig.colorbar(cf, ax=ax)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')

    # Seed points on two diagonals 
    # diagonal 1
    N1 = 20
    t1 = np.linspace(0, 1, N1)
    x1 = x1d.min() + t1*(x1d.max() - x1d.min())
    y1 = y1d.min() + t1*(y1d.max() - y1d.min())
    seeds1 = np.column_stack((x1, y1))

    # diagonal 2
    N2 = 15
    t2 = np.linspace(0, 1, N2)
    x2 = x1d.min() + t2*(x1d.max() - x1d.min())
    y2 = y1d.max() - t2*(y1d.max() - y1d.min())
    seeds2 = np.column_stack((x2, y2))

    # Add more points near the bottom corners
    # bottom‐left cluster
    Nc1 = 20
    x_c1 = np.linspace(0.0, 0.2, Nc1)
    y_c1 = np.copy(x_c1)
    seeds_c1 = np.column_stack((x_c1, y_c1))

    # bottom‐right cluster
    Nc2 = 20
    x_c2 = np.linspace(0.8, 1.0, Nc2)
    y_c2 = np.copy(x_c1)
    seeds_c2 = np.column_stack((x_c2, y_c2))

    # combine all seeds
    seeds = np.vstack((seeds1, seeds2, seeds_c1, seeds_c2))

    # streamplot with combined seeds
    ax.streamplot(
        X, Y, u, v,
        start_points=seeds,
        color='k',
        linewidth=0.5,
        arrowstyle='->',
        arrowsize=1
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout(pad=0)
    fig.savefig(f'Re_{Re_vec[i]}.png', transparent=False, bbox_inches='tight')
    plt.close(fig)
