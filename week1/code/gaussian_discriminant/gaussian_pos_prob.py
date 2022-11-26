import numpy as np

def gaussian_pos_prob(X, Mu, Sigma, Phi):
    '''
    GAUSSIAN_POS_PROB Posterior probability of GDA.
    Compute the posterior probability of given N data points X
    using Gaussian Discriminant Analysis where the K gaussian distributions
    are specified by Mu, Sigma and Phi.
    Inputs:
        'X'     - M-by-N numpy array, N data points of dimension M.
        'Mu'    - M-by-K numpy array, mean of K Gaussian distributions.
        'Sigma' - M-by-M-by-K  numpy array (yes, a 3D matrix), variance matrix of
                  K Gaussian distributions.
        'Phi'   - 1-by-K  numpy array, prior of K Gaussian distributions.
    Outputs:
        'p'     - N-by-K  numpy array, posterior probability of N data points
                with in K Gaussian distribsubplots_adjustutions.
    ''' 
    N = X.shape[1]
    K = Phi.shape[0]
    p = np.zeros((N, K))
    det = np.linalg.det(Sigma)
    inv = np.matrix(np.linalg.inv(Sigma))
    x_mu = np.matrix(X - Mu).T
    Const = 1 / (((2 * np.pi) * (det ** (1 / 2)))
    Exp = -0.5 * x_mu.T * inv * x_mu
    return float(Const * np.exp(Exp))
    
