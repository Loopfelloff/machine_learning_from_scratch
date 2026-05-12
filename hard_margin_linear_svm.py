from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt


t0, t1 = 5, 50 
def learning_rate(t):
    return t0 /(t + t1)

class StandardScaling():
    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        return self
    def transform(self,X):
        return (X-self.mean_)/self.std_

class SVM():
    def fit(self , X , y):
        np.random.seed(42)
        self.theta_ = np.array([[2] , [2] , [2]]) 
        for _ in range(1000):
            


        return self

def main():
    iris = load_iris(as_frame = True)
    X = iris.data[["petal length (cm)", "petal width (cm)"]].values
    y = iris['target'].values == 0 

    std_scaling = StandardScaling()
    std_scaling.fit(X)
    X = std_scaling.transform(X) 

    ############### Plotting Purposes #########################
    plt.plot(X[ y == True , 0 ],  X[y == True , 1] , "o")
    plt.plot(X[ y == False , 0 ],  X[y == False , 1] , "ro")
    plt.xlabel("petal length (cm)")
    plt.ylabel("petal width (cm)")
    plt.show()
main()


