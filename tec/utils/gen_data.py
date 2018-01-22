import numpy as np


def k_means_ds(idx):
    datasets = (k_means_data0, k_means_data1, k_means_data2, k_means_data3)
    return datasets[idx]()


def k_means_data0():
    x1 = np.random.multivariate_normal(mean=[-5, -5], cov=[[1, 0], [0, 1]], size=1000)
    x2 = np.random.multivariate_normal(mean=[5, 5], cov=[[1, 0], [0, 1]], size=1000)
    x3 = np.random.multivariate_normal(mean=[5, -5], cov=[[1, 0], [0, 1]], size=1000)

    x = np.r_[x1, x2, x3]
    np.random.shuffle(x)
    return x


def k_means_data1():
    x1 = np.random.multivariate_normal(mean=[-5, -5], cov=[[1, 0], [0, 1]], size=100)
    x2 = np.random.multivariate_normal(mean=[5, 5], cov=[[1, 0], [0, 1]], size=100)
    x3 = np.random.multivariate_normal(mean=[5, -5], cov=[[1, 0], [0, 1]], size=100)
    x4 = np.random.multivariate_normal(mean=[-5, 5], cov=[[1, 0], [0, 1]], size=100)
    x5 = np.random.multivariate_normal(mean=[0, 0], cov=[[1, 0], [0, 1]], size=100)

    x = np.r_[x1, x2, x3, x4, x5]
    np.random.shuffle(x)
    return x


def k_means_data2():
    x1 = np.random.multivariate_normal(mean=[-5, -5], cov=[[1, -2], [-2, 5]], size=1000)
    x2 = np.random.multivariate_normal(mean=[5, 5], cov=[[2, 0.75], [0.75, 2]], size=1000)
    x3 = np.random.multivariate_normal(mean=[5, -5], cov=[[2, -1], [-1, 2]], size=1000)

    x = np.r_[x1, x2, x3]
    np.random.shuffle(x)
    return x


def k_means_data3():
    x1 = np.random.multivariate_normal(mean=[0, -1], cov=[[5, 0], [0, 5]], size=2000)
    x2 = np.random.multivariate_normal(mean=[-5, 5], cov=[[1, 0], [0, 1]], size=500)
    x3 = np.random.multivariate_normal(mean=[5, 5], cov=[[1, 0], [0, 1]], size=500)

    x = np.r_[x1, x2, x3]
    np.random.shuffle(x)
    return x
