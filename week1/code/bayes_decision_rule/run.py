import scipy.io as sio
import numpy as np
from loguru import logger
import matplotlib.pyplot as plt
from likelihood import likelihood
from get_x_distribution import get_x_distribution
from posterior import posterior

"""
INPUT: judge_matrix: likelihood or posterior
       x_data: sample 1-By-N
       data_range: offset the start
       real_label: real label
OUTPUT: the error num of classification
"""
def classification(judge_matrix, x_data, data_range, real_label = 0):
    error_num = 0
    right_num = 0
    for i in range(x_data.shape[1]):
        x = x_data[0][i]
        select_label = np.argmax(judge_matrix[:, x - data_range[0]])
        if real_label == select_label:
            right_num += 1
        else:
            error_num += 1
    return error_num
        
def main():
    # 1. load data from data.mat
    data = sio.loadmat("data.mat")
    x1_train, x1_test, x2_train, x2_test = data["x1_train"], data["x1_test"], data["x2_train"], data["x2_test"]
    all_x = np.concatenate([x1_train, x1_test, x2_train, x2_test], 1)
    logger.info("Sample num sum: {}".format(all_x.shape[-1]))
    logger.info("Test num sum: {}".format(x1_test.shape[-1] + x2_test.shape[-1]))
    data_range = [np.min(all_x), np.max(all_x)]
    # 2. get x distribution
    train_x = get_x_distribution(x1_train, x2_train, data_range)
    test_x = get_x_distribution(x1_test, x2_test, data_range)
    # 3. calculate likelihood
    l = likelihood(train_x)
    # 4. use maix likelihood method to classification
    error_test = classification(l, x1_test, data_range, 0) + \
                 classification(l, x2_test, data_range, 1)
    # error_train = classification(l, x1_test, data_range, 0) + \
    #               classification(l, x2_test, data_range, 1)
    logger.info("likelihood: error_test: {}".format(error_test))
    # 5. calculate posterior
    p = posterior(train_x)
    error_test = classification(p, x1_test, data_range, 0) + \
                 classification(p, x2_test, data_range, 1)
    logger.info("posterior: error_test: {}".format(error_test))
    # 6. calculate risk
    risk_matrix = np.array([[0, 1], [2, 0]])
    risk = classification(p, x1_test, data_range, 0) * risk_matrix[0][1] + \
           classification(p, x2_test, data_range, 1) * risk_matrix[1][0]
    logger.info("min risk: {}".format(risk))

if __name__ == "__main__":
    main()
