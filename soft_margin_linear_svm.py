from matplotlib.colors import ListedColormap
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt


t0, t1 = 5, 50 
def learning_rate(t):
    return t0 /(t + t1)

def add_dummy_feature(arr):
    return np.c_[np.ones(len(arr)), arr]

def accuracy(y_train ,y_test):
    equal_val = y_train == y_test
    label_name , label_count = np.unique(equal_val , return_counts=True)
    if(len(label_count) <= 1):
        if label_name[0] == False:
            return 0
        elif label_name[0] == True:
            return 1
    return (label_count[1] / (label_count[0] + label_count[1]))


class TrainTestSplit():
    def __init__(self, X , y ,train_size ,random_state):
        self.X = X
        self.y = y
        self.random_state = random_state 
        self.train_size =  np.float64(train_size)
    def do_splitting(self):
        np.random.seed(self.random_state)
        self.random_shuffle = np.random.permutation(len(self.X))
        self.train_size_length = (self.train_size * len(self.X) ).astype('int64') 
        return self.X[self.random_shuffle][:self.train_size_length] , self.X[self.random_shuffle][self.train_size_length : ] , self.y[self.random_shuffle][:self.train_size_length] , self.y[self.random_shuffle][self.train_size_length : ] , self.random_shuffle

class StandardScaling():
    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        return self
    def transform(self,X):
        return (X-self.mean_)/self.std_

class SVM():
    def __init__(self , C):
        self.c_ = C

    def fit(self , X , y):
        np.random.seed(42)
        self.theta_ = np.random.rand(2,1)
        np.random.seed(42)
        self.intercept_ = np.random.rand(1,1)
        eta = 0.1
        y_b = y.astype("int8").copy()
        y_b[y_b == 0] = -1
        total_data = len(X)

        for ep in range(1000):
            for iteration in range(total_data):
                random_val = np.random.randint(total_data)
                Xi_b = X[random_val : random_val + 1]
                Yi = y_b[random_val : random_val + 1] 
                dot_prod = (np.dot(Xi_b , self.theta_) + self.intercept_).reshape(-1)
                predict_val = (dot_prod * Yi[0])[0] 
                eta = learning_rate(ep * total_data + iteration)
                if predict_val >= 1 :
                    self.theta_ = self.theta_ - eta * self.theta_
                else:
                    self.theta_  = self.theta_ - eta * (self.theta_ - self.c_ * (Xi_b.T * Yi[0]))
                    self.intercept_ = self.intercept_ + eta * self.c_ * Yi[0]
        return self
    
    def predict(self, X):
        self.predict_proba_ = (np.dot(X , self.theta_) + self.intercept_).reshape(-1)
        return self.predict_proba_ >= 0



def main():
    iris = load_iris(as_frame = True)
    X = iris.data[["petal length (cm)", "petal width (cm)"]].values
    y = iris['target'].values == 0 
    train_test_split = TrainTestSplit(X, y , 0.8 ,  42)
    X_train , X_test , y_train , y_test , _ = train_test_split.do_splitting()
    std_scaling = StandardScaling()
    std_scaling.fit(X_train)
    X_train = std_scaling.transform(X_train) 

    svm = SVM(C=100)

    svm.fit(X_train , y_train)

    X_test = std_scaling.transform(X_test)
    y_predict = svm.predict(X_test)

    accuracy_val = accuracy(y_test , y_predict)

    print(f"The accuracy is : {accuracy_val}")
    print("The model parameters are")
    print(svm.theta_)
    print(svm.intercept_)

    ############### Plotting Purposes #########################

    custom_cmap = ListedColormap(["#fafab0", "#9898ff", "#a0faa0"])

    x0 , x1 = np.meshgrid(
            np.linspace(0,8,500).reshape(-1,1),
            np.linspace(0,3.5, 200).reshape(-1,1),
            )
    X_new = np.c_[x0.ravel(), x1.ravel()]
    X_new = std_scaling.transform(X_new)
    y_predict_plot = svm.predict(X_new)
    y_predict_plot_proba = svm.predict_proba_ 
    zz1 = y_predict_plot_proba.reshape(x0.shape)
    zz = y_predict_plot.reshape(x0.shape)
    plt.figure(figsize=(10, 4))
    plt.contourf(x0, x1, zz, cmap=custom_cmap)
    plt.contour(x0, x1, zz1, cmap="hot", levels=[-1 , 0 , 1])
    plt.xlabel("Petal length")
    plt.ylabel("Petal width")
    plt.axis([0.5, 7, 0, 3.5])
    plt.grid()
    ##################################################
    plt.plot(X[ y == True , 0 ],  X[y == True , 1] , "o")
    plt.plot(X[ y == False , 0 ],  X[y == False , 1] , "ro")
    plt.xlabel("petal length (cm)")
    plt.ylabel("petal width (cm)")
    
    plt.show()
main()


