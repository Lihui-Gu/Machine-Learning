# some basic imports
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
from loguru import logger
from likelihood import likelihood
logger.add("file_{time}.log") 

def classification(judge_matrix, x_data, real_label):
    right_num = 0
    error_num = 0
    feature_n = x_data.shape[-1]
    logger.info("total feature is {}".format(feature_n))
    for i in range(x_data.shape[0]):
        if i % 10 == 0:
            logger.info("Test step in {} patch, right num {}, error num {}.".format(i, right_num, error_num))
        x = x_data[i]
        value_0 = 0
        value_1 = 0
        for j in range(feature_n):            
            if x[0, j] < 0:
                logger.error("data error")
            elif x[0, j] == 0:
                value_0 += np.log(1 - judge_matrix[0][j])
                value_1 += np.log(1 - judge_matrix[1][j])
            else:
                value_0 += np.log(judge_matrix[0][j]) * x[0, j]
                value_1 += np.log(judge_matrix[1][j]) * x[0, j]
        if(value_0 > value_1):
            select_label = 0
        else:
            select_label = 1
        if(select_label == real_label):
            right_num += 1
        else:
            error_num += 1
    logger.info("Test finish, right num is {}, error num is {}".format(right_num, error_num))


def main():
    # ham_train contains the occurrences of each word in ham emails. 1-by-N vector
    ham_train = np.loadtxt('ham_train.csv', delimiter=',')
    # spam_train contains the occurrences of each word in spam emails. 1-by-N vector
    spam_train = np.loadtxt('spam_train.csv', delimiter=',')
    x_train = np.concatenate([ham_train[np.newaxis], spam_train[np.newaxis]], 0)
    l = likelihood(x_train)
    # N is the size of vocabulary.
    N = ham_train.shape[0]
    # There 9034 ham emails and 3372 spam emails in the training samples
    num_ham_train = 9034
    num_spam_train = 3372
    x = np.vstack([ham_train, spam_train]) + 1
    # ham_test contains the occurences of each word in each ham test email. P-by-N vector, with P is number of ham test emails.
    i, j, ham_test = np.loadtxt('ham_test.txt').T
    i = i.astype(int)
    j = j.astype(int)
    ham_test_tight = scipy.sparse.coo_matrix((ham_test, (i - 1, j - 1)))
    ham_test = scipy.sparse.csr_matrix((ham_test_tight.shape[0], ham_train.shape[0]))
    ham_test[:, 0:ham_test_tight.shape[1]] = ham_test_tight
    logger.info("Test Ham Email.")
    classification(l, ham_test, 0)

    # spam_test contains the occurences of each word in each spam test email. Q-by-N vector, with Q is number of spam test emails.
    i, j, spam_test = np.loadtxt('spam_test.txt').T
    i = i.astype(int)
    j = j.astype(int)
    spam_test_tight = scipy.sparse.csr_matrix((spam_test, (i - 1, j - 1)))
    spam_test = scipy.sparse.csr_matrix((spam_test_tight.shape[0], spam_train.shape[0]))
    spam_test[:, 0:spam_test_tight.shape[1]] = spam_test_tight
    logger.info("Test Spam Email.")
    classification(l, spam_test, 1)

if __name__ == "__main__":
    main()
