import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

t0 , t1 = 5, 50
def add_dummy_feature(arr):
    return np.array([[1 , *item] for item in arr ])
def learning_rate(t):
    return t0/(t1 + t)

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
    for ep in range(epoch):
        for iteration in range(total_data):
            random_val = np.random.randint(total_data)
            Xi_b = X_b[random_val : random_val+1]
            Yi = Y[random_val : random_val+1]
            gradients = 2 * Xi_b.T @ (Xi_b @ theta - Yi)
            eta = learning_rate(ep * total_data + iteration)
            theta = theta - eta * gradients
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
    plt.title("Convergence of the weights")
    plt.plot(theta_0, theta_1 , '-o')
    plt.show()

if __name__ == "__main__":
    main(3,4)
