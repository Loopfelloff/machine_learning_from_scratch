# this is where i am gonna focus on
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def add_dummy_feature(arr):
    return np.array([[1 , *item] for item in arr ])

def ten_degree_polynomialize(X):
    return np.array([(*i , *i**2 , *i**3, *i**4 , *i**5, *i**6 , *i**7, *i**8, *i**9, *i**10) for i in X ])

def main():
    m = 20
    X = np.linspace(-3 , 3, m).reshape([-1,1])
    X_transformed = add_dummy_feature(ten_degree_polynomialize(X))
    _ , n = X_transformed.shape
    A = np.identity(n)
    A[0 , 0] = 0 # to include for bias i guess
    Y = 4 * X +3 + 3 * np.random.randn(m,1)
    params = (np.linalg.inv(X_transformed.T @ X_transformed) @ X_transformed.T @ Y)
    regularized_param = (np.linalg.inv(X_transformed.T @ X_transformed + 1 * A))@X_transformed.T @ Y
    regularized_param_1 = (np.linalg.inv(X_transformed.T @ X_transformed + 1000 * A))@X_transformed.T @ Y
    predicted_Y = X_transformed @ params
    regularized_Y = X_transformed @ regularized_param
    regularized_Y_1 = X_transformed @ regularized_param_1
    plt.plot(X , Y, 'o')
    plt.plot(X , predicted_Y , 'r' , label="Overfitting")
    plt.plot(X , regularized_Y , 'g' , label="Regularized")
    plt.plot(X , regularized_Y_1 , 'm' , label="Heavy Regularization")
    plt.legend(loc="upper right")
    plt.show()
    



if __name__ == "__main__":
    main()
