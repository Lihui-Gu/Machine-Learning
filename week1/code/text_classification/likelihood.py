import numpy as np

def likelihood(x):
    '''
    LIKELIHOOD Different Class Feature Liklihood 
    INPUT:  x, features of different class, C-By-N numpy array
            C is the number of classes, N is the number of different feature

    OUTPUT: l,  likelihood of each feature(from smallest feature to biggest feature) given by each class, C-By-N numpy array
    just calculate the p(x|Ci)
    '''
    C, N = x.shape
    l = np.zeros((C, N))
    # x_sum shape C-By-1
    x_sum = x.sum(axis=1)
    #TODO calculate likelihood
    for i in range(C):
        l[i, :] = x[i, :] / x_sum[i]
    return l
