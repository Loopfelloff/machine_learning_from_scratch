import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# using normal equation the minimal value is basically.

# theta = X^tY(X^tX)^-1 also called the normal equation or closed form method. 
# later on gradient descent uses iterative approach to solve the problem.
def add_dummy_feature(arr):
    return np.array([[1 , *item] for item in arr ])

def main(slope , intercept):
    m = 1000
    X = np.random.rand(m , 1) * 2
    Y = intercept + slope * X + np.random.randn(m,1)
    plt.plot(X, Y , "o")
    plt.plot([0,2] , [4, 10], 'g' , label="True Curve")
    X_b = add_dummy_feature(X)  
    params = (np.linalg.inv(X_b.T @ X_b) @ X_b.T @ Y)
    slope_predicted , intercept_predicted = params[0 , 0] , params[1, 0]
    initial_val = [0,2]
    final_val  =  [intercept_predicted + slope_predicted * initial_val[0],
                   intercept_predicted + slope_predicted * initial_val[1],
]
    plt.plot(initial_val , final_val, 'r' , label="Predicted Curve")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.legend(["Data" , "True Curve" , "Predicted Curve"])
    plt.show()

if __name__ == "__main__":
    main(3,4)
