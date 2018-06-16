from time import time
import numpy as np
from sklearn.linear_model import Lasso

def test_lasso(n_points= 100000, n_dims = 100):

    x = np.random.normal(size=(n_points, n_dims)).reshape((n_points, n_dims))
    y = np.dot(x, np.linspace(-1, 1, n_dims)) + np.random.normal(size=(n_points))

    lasso = Lasso(alpha=1.0)

    start = time()
    lasso.fit(x,y)
    end = time()

    return end-start


def test_ridge(n_points= 10000, n_dims = 100):
    times = []
    for i in range(5):
        x = np.random.normal(size=(n_points, n_dims)).reshape((n_points, n_dims))
        y = np.dot(x, np.linspace(-1, 1, n_dims)) + np.random.normal(size=(n_points))

        lasso = Lasso(alpha=0.1)

        start = time()
        lasso.fit(x,y)
        end = time()

        times.append(time)

    return np.mean(times)



for n_dims in [10, 100, 1000, 5000]:
    print( "Dimensions: {}\tExecution time: {}".format(n_dims, test_lasso(n_dims=n_dims)) )
