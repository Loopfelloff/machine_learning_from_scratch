import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def add_dummy_feature(arr):
    return np.array([[1 , *item] for item in arr ])

def main(slope , intercept):
    total_data = 1000
    epoch = 1000
    theta_0 = []
    theta_1 = []
    X = np.random.rand(total_data , 1) * 2
    Y = intercept + slope * X + np.random.randn(total_data,1)
    plt.plot(X, Y , "o")
    plt.plot([0,2] , [4, 10], 'g' , label="True Curve")
    X_b = add_dummy_feature(X)  
    np.random.seed(42)
    theta = np.random.randn(2,1)
    for _ in range(0, epoch):
        theta = theta - 0.1 * 2/total_data * X_b.T @ (X_b@theta -Y)
        theta_0.append(theta[0,0])
        theta_1.append(theta[1,0])
    slope_predicted , intercept_predicted = theta[0 , 0] , theta[1, 0]
    initial_val = [0,2]
    final_val  =  [intercept_predicted + slope_predicted * initial_val[0],
                   intercept_predicted + slope_predicted * initial_val[1],
]
    plt.plot(initial_val , final_val, 'r' , label="Predicted Curve")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.legend(["Data" , "True Curve" , "Predicted Curve"])
    plt.show()
    plt.plot(theta_0, theta_1 , '-o')
    plt.title("Convergence of the weights")
    plt.show()

if __name__ == "__main__":
    main(3,4)
