import numpy as np
from scipy.linalg import eigh
from scipy.special import expit
import numpy as np
from scipy.linalg import qr
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
import tensorly as tl
# np.random.seed(42)

def matlab_eigs(M, k):
    """
    Find the largest k eigenvalues and their corresponding eigenvectors.
    
    Args:
    M (numpy.ndarray): Input matrix.
    k (int): Number of eigenvalues and eigenvectors to find.

    Returns:
    V (numpy.ndarray): Eigenvectors corresponding to the largest k eigenvalues.
    D (numpy.ndarray): Largest k eigenvalues.
    """
    # Find the k largest eigenvalues and their corresponding eigenvectors
    D, V = eigsh(M, k=k, which='LM')

    # Sort the eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(abs(D))[::-1]
    D = D[sorted_indices[:k]]
    V = V[:, sorted_indices[:k]]

    return V, D

def calculate_gradient(X, y, W, sig, diff_sigma):
    a = 1 / W.shape[0]
    z = activation(W, X, sig)
    outputs = a * z
    r = np.sum(outputs.T, axis=1) - y + 5 # why plus 5
    g = np.sum(r * z, axis=1)
    return g

def activation(W, X, sig):
    z = sig(W @ X)
    z = np.array(z)
    return z

def qrj1d(X, mode='B', err_iter=1e-4, rbalance=3):
    n, m = X.shape
    N = m // n

    ERR = 1 * 10**-4
    RBALANCE = 3
    ITER = 200
    MODE = mode.upper()

    if MODE == 'B':
        ERR = ERR * n

    JJ = []
    EERR = []
    EERRJ2 = []

    X1 = X.copy()
    B = np.eye(n)
    J = 0

    for t in range(N):
        J += np.linalg.norm(X1[:, t * n:(t + 1) * n] - np.diag(np.diag(X[:, t * n:(t + 1) * n])), 'fro') ** 2
    JJ.append(J)

    err = ERR * n + 1
    k = 0
    while err > ERR and k < ITER:
        k += 1
    
        L = np.eye(n)
        U = np.eye(n)

        for i in range(1, n):
            for j in range(i):
                G = np.vstack((-X[i, i:m:n] + X[j, j:m:n], -2 * X[i, j:m:n]))
                U1, D1, V1 = np.linalg.svd(G @ G.T)
                v = U1[:, 0]
                theta = 0.5 * np.arctan(v[1] / v[0])
                c = np.cos(theta)
                s = np.sin(theta)
                h1 = c * X[:, j:m:n] - s * X[:, i:m:n]
                h2 = c * X[:, i:m:n] + s * X[:, j:m:n]
                X[:, j:m:n] = h1
                X[:, i:m:n] = h2
                h1 = c * X[j, :] - s * X[i, :]
                h2 = s * X[j, :] + c * X[i, :]
                X[j, :] = h1
                X[i, :] = h2
                h1 = c * U[j, :] - s * U[i, :]
                h2 = s * U[j, :] + c * U[i, :]
                U[j, :] = h1
                U[i, :] = h2

        for i in range(n):
            for j in range(i + 1, n):
                cindex = np.arange(m)
                cindex = np.delete(cindex, np.arange(j, m, n))
                a = -(X[i, cindex] @ X[j, cindex].T) / (X[i, cindex] @ X[i, cindex].T)
                if np.abs(a) > 1:
                    a = np.sign(a) * 1
                X[j, :] = a * X[i, :] + X[j, :]
                I = np.arange(i, m, n)
                J = np.arange(j, m, n)
                X[:, J] = a * X[:, I] + X[:, J]
                L[j, :] = L[j, :] + a * L[i, :]

        B = L @ U @ B
        err = np.max(np.abs(L @ U - np.eye(n)))
        EERR.append(err)

        if k % RBALANCE == 0:
            d = np.sum(np.abs(X), axis=1)
            D = np.diag(1.0 / d * N)
            Dinv = np.diag(d * N)
            for t in range(N):
                X[:, t * n:(t + 1) * n] = D @ X[:, t * n:(t + 1) * n] @ D
            B = D @ B

        BB = B.copy()
        Binv = np.linalg.inv(B)
        J = 0
        for t in range(N):
            J += np.linalg.norm(X1[:, t * n:(t + 1) * n] - Binv @ np.diag(np.diag(X[:, t * n:(t + 1) * n])) @ Binv.T, 'fro') ** 2
        JJ.append(J)

        if MODE == 'E':
            err = np.abs(JJ[-2] - JJ[-1]) / JJ[-2]
            EERRJ2.append(err)

    Y = X
    S = {'iterations': k, 'LUerror': EERR, 'J2error': JJ, 'J2RelativeError': EERRJ2}

    return Y, BB, S, 

def no_tenfact(T, L, k):
    p = T.shape[0]
    sweeps = [0, 0]

    M = np.zeros((p, p * L))
    W = np.zeros((p, L))

    for l in range(L):
        W[:, l] = np.random.randn(p)
        # W[:, l] = np.array([-0.5382, 0.8672])
        W[:, l] = W[:, l] / np.linalg.norm(W[:, l])
        M[:, (l * p):(l * p + p)] = tl.tenalg.multi_mode_dot(T, [np.eye(p), np.eye(p), W[:, l].T], transpose=True)

    D, U, S = qrj1d(M)

    Ui = np.linalg.inv(U)

    Ui_norms = np.sqrt(np.sum(Ui ** 2, axis=1))
    Ui_normalized = Ui / Ui_norms[:, np.newaxis]

    dot_products = Ui_normalized.T @ W
    Lambdas = np.zeros((p, L))
    for l in range(L):
        Lambdas[:, l] = (np.diag(D[:, (l * p):(l * p + p)]) / dot_products[:, l]) * (Ui_norms ** 2)

    idx0 = np.argsort(np.mean(np.abs(Lambdas), axis=1), axis=0)[::-1]
    Lambda0 = np.mean(Lambdas[idx0[:k], :], axis=1)
    V = Ui_normalized[:, idx0[:k]]

    sweeps[0] = S['iterations']
    sweeps[1] = S['iterations']

    W = Ui_normalized
    M = np.zeros((p, p * W.shape[1]))

    for l in range(W.shape[1]):
        w = W[:, l]
        w = w / np.linalg.norm(w)

        M[:, (l * p):(l * p + p)] = tl.tenalg.multi_mode_dot(T, [np.eye(p), np.eye(p), w.T], transpose=True)

    D, U, S = qrj1d(M)
    Ui = np.linalg.inv(U)
    Ui_norm = Ui / np.sqrt(np.sum(Ui ** 2, axis=0))[:, np.newaxis]
    V1 = Ui_norm
    sweeps[1] += S['iterations']

    Lambda = np.zeros((p, 1))
    for l in range(p):
        Z = np.linalg.inv(V1)
        X = Z @ M[:, (l * p):(l * p + p)] @ Z.T
        Lambda = Lambda + np.abs(np.diag(X))[:, np.newaxis]

    idx = np.argsort(np.abs(Lambda), axis=0)[::-1]
    V1 = Ui_norm[:, idx[:k].flatten()]

    misc = {
        'V0': V,
        'sweeps': sweeps
    }

    return V1, Lambda, misc

def sig(x):
    return x**3 + x**2

def diff_sig(x):
    return 3*x**2 + 2*x

def main():
    for d in range(50, 300, 50):
        d = 10
        B = 2
        m = 768

        X = np.eye(d, B)
        # X = np.abs(np.array([
        #  [-1.5902e+00,  6.4896e-01, -6.6215e-01, -6.6466e-01,  4.8475e-01,
        #   5.2235e-01,  7.6160e-02,  2.6702e-01, -3.9233e-01,  6.6559e-01,
        #  -6.9282e-01, -1.3897e+00, -1.2413e+00,  7.6128e-01, -1.0343e+00,
        #  -2.1138e-01, -1.7131e-02, -1.6605e-02, -1.7239e-01,  1.0149e+00,
        #  -6.1354e-01, -9.4482e-02, -8.1964e-01,  6.9638e-01,  1.1052e+00,
        #  -2.4838e-01, -4.7157e-01,  5.9361e-01,  3.9256e-01,  3.0840e-01,
        #   6.4800e-01,  1.2831e+00, -7.6369e-01, -2.4490e-01, -7.1711e-01,
        #  -1.0530e+00,  1.2676e+00, -8.7227e-01, -3.8455e-01,  1.8136e+00,
        #   3.0313e-01, -1.3318e+00,  1.2053e+00, -3.6410e-01,  1.4545e-01,
        #  -5.3366e-01, -1.7756e+00,  1.0870e+00, -5.8663e-01, -6.4841e-01],
        # [-1.6801e+00,  1.1927e-01, -8.1798e-01,  1.8851e-02,  1.1044e+00,
        #   4.3372e-01,  1.4487e-01,  4.2379e-02, -9.4255e-01,  5.7226e-01,
        #  -6.5297e-01, -1.0227e-01, -1.0915e+00,  1.6561e+00,  7.7471e-04,
        #   4.9941e-01,  1.0521e-02,  2.0080e-03, -3.4772e-01,  1.8644e-01,
        #  -6.5687e-01, -5.4844e-01, -4.2292e-01,  6.8358e-01,  6.8575e-01,
        #   1.9123e-01, -6.0223e-01,  6.6301e-01,  3.5389e-01,  4.1500e-01,
        #  -3.4312e-01,  2.3895e-01, -1.0154e-01, -2.4533e-01, -6.5648e-01,
        #  -1.0809e+00,  1.3587e+00, -4.7886e-01, -3.2871e-01,  7.2975e-01,
        #   6.4553e-01, -1.3983e+00,  1.0700e+00, -4.2421e-01,  6.2344e-02,
        #  -1.8363e-01, -1.4588e+00,  7.2451e-01, -8.7547e-01, -2.6528e-01]
        # ]).reshape(50, 2))
        # X = np.round(X/np.linalg.norm(X, ord=2, axis=0), 3)
        
        from scipy.spatial import distance
        print(f"cosin similarity: {1-distance.cosine(X[:, 0].reshape(-1), X[:,1].reshape(-1))}")
        
        # X2 = np.abs(np.array([-1.5902e+00,  6.4896e-01, -6.6215e-01, -6.6466e-01,  4.8475e-01,
        #   5.2235e-01,  7.6160e-02,  2.6702e-01, -3.9233e-01,  6.6559e-01,
        #  -6.9282e-01, -1.3897e+00, -1.2413e+00,  7.6128e-01, -1.0343e+00,
        #  -2.1138e-01, -1.7131e-02, -1.6605e-02, -1.7239e-01,  1.0149e+00,
        #  -6.1354e-01, -9.4482e-02, -8.1964e-01,  6.9638e-01,  1.1052e+00,
        #  -2.4838e-01, -4.7157e-01,  5.9361e-01,  3.9256e-01,  3.0840e-01,
        #   6.4800e-01,  1.2831e+00, -7.6369e-01, -2.4490e-01, -7.1711e-01,
        #  -1.0530e+00,  1.2676e+00, -8.7227e-01, -3.8455e-01,  1.8136e+00,
        #   3.0313e-01, -1.3318e+00,  1.2053e+00, -3.6410e-01,  1.4545e-01,
        #  -5.3366e-01, -1.7756e+00,  1.0870e+00, -5.8663e-01, -6.4841e-01]).reshape(50, 1))
        # X2 = np.round(X2/np.linalg.norm(X2, ord=2, axis=0), 3)
        
        # y = 2 * np.random.randint(0, 2, (1, B)) - 1
        y = np.array([[1,-1]])

        mu = np.zeros((d, 1))
        Sigma = np.eye(d)

        W = np.random.multivariate_normal(mu.flatten(), Sigma, m)
        # W = np.array([
        #     [-0.5382,-0.5915,-0.4091],
        #     [0.8672,1.8444,-1.2819],
        #     [0.9760,1.8169,-0.2849],
        #     [0.3374,-0.1238,-0.0648],
        #     [-0.9961,-1.1106,1.0004],
        #     [-0.5232,-0.6809,-0.8013],
        #     [-1.2974,0.0142,0.0918],
        #     [0.9174,-0.0596,0.2738],
        #     [0.1766,-0.6610,-2.1335],
        #     [0.7552,0.3060,0.4040]])

        g = calculate_gradient(X, y, W, sig, diff_sig) # (10000,)

        M = np.zeros((d, d)) # W 10000 x 50
        a = np.sum(g)

        Beta = 2
        for i in range(d): #50
            for j in range(d): #50
                M[i, j] = np.sum(g * W[:, i] * W[:, j])
                if i == j:
                    M[i, i] = M[i, i] - a

        # D, V = eigh(M, eigvals=(d-Beta, d-1))
        
        V, D = matlab_eigs(M, Beta)
        WV = W @ V

        T = np.zeros((Beta, Beta, Beta))
        for i in range(Beta):
            for j in range(i, Beta):
                for k in range(j, Beta):
                    T[i, j, k] = np.sum(g * WV[:, i] * WV[:, j] * WV[:, k])
                    T[i, k, j] = T[i, j, k]
                    T[j, i, k] = T[i, j, k]
                    T[j, k, i] = T[i, j, k]
                    T[k, i, j] = T[i, j, k]
                    T[k, j, i] = T[i, j, k]

        for i in range(Beta):
            for j in range(Beta):
                a = np.sum(g * WV[:, i])
                T[i, j, j] = T[i, j, j] - a
                T[j, i, j] = T[j, i, j] - a
                T[j, j, i] = T[j, j, i] - a

        T = T / m

        print('Reconstruction starts!')

        # T = np.array([[[30.9539, 0.9502], [0.9502, 24.5770]], [[ 0.9502, 24.5770],[24.5770, -15.7356]]])

        # Please replace `no_tenfact` with the appropriate tensor factorization function
        rec_X, _, misc = no_tenfact(T, 100, B)

        new_recX = V @ rec_X

        for i in range(B):
            # if np.min(new_recX[:, i]) < -0.5:
            new_recX[:, i] = np.abs(new_recX[:, i])
        
        new_recX = np.round(new_recX / np.linalg.norm(new_recX, ord=2, axis=0), 3)

        from scipy.spatial import distance
        print(f"cosin similarity: {1-distance.cosine(new_recX.reshape(-1), X.reshape(-1))}")
        print(f"normalized error: {np.linalg.norm(new_recX - X, ord=2, axis=0)}" )
        
        b = new_recX[:, 1].copy()
        new_recX[:, 1] = new_recX[:, 0]
        new_recX[:, 0] = b
        print(f"cosin similarity: {1-distance.cosine(new_recX.reshape(-1), X.reshape(-1))}") 
        print(f"normalized error: {np.linalg.norm(new_recX - X, ord=2, axis=0)}" )
        break
        
if __name__ == "__main__":
    main()