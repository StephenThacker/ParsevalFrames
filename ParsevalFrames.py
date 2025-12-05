import numpy as np
from scipy.optimize import minimize, NonlinearConstraint, Bounds
from scipy.linalg import svd


def pframe(Q):
    """Construct a Parseval frame from Q."""
    _, N = Q.shape
    U, s, Vh = svd(Q, full_matrices=False)
    #s returns only the singular values, which can have
    #less dimension than the actual singular value
    print("Vh", Vh)
    print("s")
    print(s)
    ssq = s**2
    print("ssq", ssq)
    print("last svd value", s[-1])

    sigma_sq = np.zeros(N)
    sigma_sq[:len(ssq)] = ssq

    sigma2_diag = np.sqrt(np.maximum(0, 1 - sigma_sq))
    sigma2_diag[np.abs(sigma2_diag) <= 1e-6] = 0

    Sigma2 = np.diag(sigma2_diag)
    V = Vh.T
    D2 = Sigma2 @ V.T
    QD2 = np.vstack((Q, D2))
    QD2 = QD2[np.any(QD2, axis=1), :]

    B = QD2 @ np.diag(Q[0, :])
    B[np.abs(B) <= 1e-6] = 0
    return B


def obj_function(lambd, Q):
    lambd_full = np.concatenate(([1.], lambd))
    D = np.diag(lambd_full)
    M = D @ Q
    return -np.trace(M.T @ M)


def jac_obj(lambd, Q):
    row_norms_sq = np.sum(Q[1:, :] ** 2, axis=1)
    return -2 * lambd * row_norms_sq


def nonlinear_con(lambd, Q):
    """Constraint: largest singular value ≤ 1."""
    lambd_full = np.concatenate(([1.], lambd))
    D = np.diag(lambd_full)
    M = D @ Q
    s = svd(M, compute_uv=False)[0]  # largest singular value
    return 1 - s


def jac_con(lambd, Q):
    lambd_full = np.concatenate(([1.], lambd))
    D = np.diag(lambd_full)
    M = D @ Q
    U, s, Vh = svd(M, full_matrices=False)

    u = U[:, 0]   # left singular vector
    v = Vh[0, :]  # right singular vector

    jac = np.zeros(len(lambd))
    for k in range(len(lambd)):
        jac[k] = u[k + 1] * np.dot(Q[k + 1, :], v)

    return -jac


def funmin(a, B1):
    """Main function to generate Parseval Framelet high-pass filters."""
    a = a.flatten()
    c = np.sqrt(a)
    '''
    print("1/c")
    print(1/c)'''

    if B1.size == 0:
        Q = c[None, :]
        print("Q")
        print(Q)
        return pframe(Q)
    '''
    print("B1")
    print(B1)
    print("diag")
    print(np.diag(1/c))'''

    D1 = B1 @ np.diag(1 / c)
    colin_vec = D1@c.T

    
    print("checking collinearity requirements")
    for i in range(0,len(colin_vec)):
        if colin_vec[i] != 0:
            print("failed for b_%s" % i)
            raise ValueError("Does not meet collinearity requirement for low pass filter. b_i * (1/c)^T != 0")
        else: 
            print("passed for b_%s" % i)
    #must be all zero vectors for D1 applied to the transpose of c
    Q = np.vstack((c, D1))

    nvars = B1.shape[0]
    #print("nvars", nvars)
    lb = np.zeros(nvars)
    ub = np.ones(nvars)

    x0 = np.ones(nvars)

    # Define wrapped functions for optimizer
    def obj(x): return obj_function(x, Q)
    def jobj(x): return jac_obj(x, Q)
    def con(x): return nonlinear_con(x, Q)
    def jcon(x): return jac_con(x, Q)

    # Trust-constr nonlinear constraint
    constraint = NonlinearConstraint(con, 0, np.inf, jac=jcon)
    bounds = Bounds(lb, ub)

    options = {
        'xtol': 1e-12,
        'gtol': 1e-12,
        'barrier_tol': 1e-12,
        'maxiter': 2000,
        'verbose': 1
    }

    # MAIN OPTIMIZATION — robust to row permutations!
    res = minimize(
        obj, x0,
        method='trust-constr',
        jac=jobj,
        hess='2-point',
        bounds=bounds,
        constraints=[constraint],
        options=options
    )

    #print("res",res)

    x = res.x
    #print("Optimization finished. Largest singular value:",
    #      svd(np.diag(np.concatenate(([1], x))) @ Q, compute_uv=False)[0])

    newQ = np.diag(np.concatenate(([1], x))) @ Q

    # Enforce top singular value = 1
    U, s, Vh = svd(newQ, full_matrices=False)
    if s[0] < 1:
        s[0] = 1
    newQ = U @ np.diag(s) @ Vh

    Bmat = pframe(newQ)
    return Bmat, x


if __name__ == "__main__":
    a = np.array([1, 2, 1, 2, 4, 2, 1, 2, 1]) / 16.0
    print("a", a)

    B1 = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0, -1],
        [0, 1, 0, 0, 0, 0, 0, -1, 0],
        [0, 0, 1, 0, 0, 0, -1, 0, 0],
        [0, 0, 0, -1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, -2, 0, 1, 0, 0],
        [0, 1, 0, 0, -2, 0, 0, 1, 0],
        [1, 0, 0, 0, -2, 0, 0, 0, 1],
        [0, 0, 0, 1, -2, 1, 0, 0, 0]
    ])

    Bmat, res = funmin(a, B1)

    print("\nBmat (scaled by 100):")
    print(np.round(Bmat * 100, 1))

    print("\nOptimized λ:")
    print(res)
