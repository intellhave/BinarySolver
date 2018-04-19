import numpy as np

def relaxed_ba_bias(Xinput, L, lamb, beta, max_iter=300):
    X = Xinput.T               # X: n_samples x n_dim
    D, m = X.shape
    B = np.sign(np.random.rand(L, m) - 0.5)
    c1 = np.random.rand(L,1)
    c2 = np.random.rand(D,1)

    for i in range(max_iter):
        # given B, compute W1
        W1 = lamb*np.matmul(np.matmul((B + c1), X.T), \
                                np.linalg.inv(lamb*np.matmul(X,X.T) + beta*np.eye(D)))

        # given B, compute W2
        W2 = np.matmul( np.matmul((X-c2), B.T), \
                       np.linalg.inv(np.matmul(B,B.T) + beta*np.eye(L)))

        # compute c1
        c1 = (1.0/m)*np.matmul(B - np.matmul(W1, X), np.ones((m,1)))
         # compute c2
        c2 = (1.0/m)*np.matmul(X - np.matmul(W2, B), np.ones((m,1)))

        # given W1, W2, c1, c2, compute B
        Xtmp = X - c2
        H = np.matmul(W1, X) + c1
        B = learn_B_new(Xtmp.T, W2.T, B.T, H.T, lamb);

        B = B.T

        # X_reconst = np.matmul(W2, np.sign(np.matmul(W1, X) + c1)) + c2
        # mse = np.mean(np.square(X_reconst - X))
        # print('mse {}'.format(mse))
    return W2, W1, c2, c1, B

def learn_B_new(Y, Wg, Bpre, XF, nu):
    B = Bpre.copy()

    Q = nu * XF + np.matmul(Y, Wg.T)
    _, L = B.shape

    for time in range(20):
        for k in range(L):  # closed form for each row of B
            Zk = np.concatenate((B[:, 0:k], B[:, k+1:]), axis=1) #  ignore bit k
            Wkk = Wg[k,:].reshape(-1, 1)
            Wk = np.concatenate((Wg[0:k, :], Wg[k+1:, :]), axis=0)
            B[:,k] = np.sign(Q[:,k] - np.matmul(Zk, np.matmul(Wk, Wkk)).reshape(-1))

    return B
