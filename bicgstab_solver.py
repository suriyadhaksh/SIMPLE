import numpy as np

def bicgstab(A, b, x0=None, rt0=None, max_iter=1000, tol=1e-8):
    """
    BiCGSTAB solver for A x = b.

    Parameters
    ----------
    A : {array_like or sparse}
        n×n matrix or linear operator supporting .dot(x)
    b : array_like, shape (n,)
        Right‑hand side vector.
    x0 : array_like, shape (n,), optional
        Initial guess. Defaults to zero vector.
    rt0 : array_like, shape (n,), optional
        Shadow residual. If None, set rt0 = r0 = b - A.dot(x0).
    max_iter : int, optional
        Maximum number of iterations (default 1000).
    tol : float, optional
        Relative tolerance on residual norm (default 1e-8).

    Returns
    -------
    x : ndarray, shape (n,)
        Approximate solution.
    converged : bool
        False if solver met tolerance before max_iter.
    iters : int
        Number of iterations performed.
    res_hist : list of float
        ‖rₖ‖₂ at each iteration (including initial).
    """
    b = np.asarray(b)
    n = b.size

    # Initial guess
    if x0 is None:
        x = np.zeros(n, dtype=b.dtype)
    else:
        x = np.array(x0, dtype=b.dtype)

    # Initial residuals
    r = b - A.dot(x)
    r_hat = r.copy() if rt0 is None else np.array(rt0, dtype=b.dtype)

    norm_b = np.linalg.norm(b)
    if norm_b == 0:
        norm_b = 1.0

    tol_threshold = tol * norm_b
    res_norm = np.linalg.norm(r)
    res_hist = [res_norm]

    # Check initial convergence
    if res_norm <= tol_threshold:
        return x, False, 0, res_hist

    # Algorithmic scalars
    rho_old = alpha = omega = 1.0
    v = np.zeros(n, dtype=b.dtype)
    p = np.zeros(n, dtype=b.dtype)

    for k in range(1, max_iter + 1):
        rho_new = np.dot(r_hat, r)
        if rho_new == 0:
            # Breakdown
            break

        if k == 1:
            p[:] = r
        else:
            beta = (rho_new / rho_old) * (alpha / omega)
            p[:] = r + beta * (p - omega * v)

        v = A.dot(p)
        alpha = rho_new / np.dot(r_hat, v)
        s = r - alpha * v

        s_norm = np.linalg.norm(s)
        if s_norm <= tol_threshold:
            x += alpha * p
            res_hist.append(s_norm)
            return x, False, k, res_hist

        t = A.dot(s)
        tt = np.dot(t, t)
        if tt == 0:
            # Breakdown
            break

        omega = np.dot(t, s) / tt
        x += alpha * p + omega * s
        r = s - omega * t

        res_norm = np.linalg.norm(r)
        res_hist.append(res_norm)

        if res_norm <= tol_threshold:
            return x, False, k, res_hist

        if omega == 0:
            # Breakdown
            break

        rho_old = rho_new

    # If we exit the loop without meeting tol
    return x, True, k, res_hist
