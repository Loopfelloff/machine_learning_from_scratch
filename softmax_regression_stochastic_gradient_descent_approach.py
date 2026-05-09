from operator import add
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt

t0, t1 = 5, 50 

def add_dummy_feature(arr):
    return np.c_[np.ones(len(arr)), arr]
def learning_rate(t):
    return t0 /(t + t1)

def accuracy(y_true , y_pred):
    check_arr = (y_true == y_pred)
    values , count = np.unique(check_arr , return_counts=True)
    first_tag_accuracy = count[0] / (count[0] + count[1])
    second_tag_accuracy = count[1] / (count[0] + count[1])
    print(values)
    if values[0] == True:
        return first_tag_accuracy
    else :
        return second_tag_accuracy

class StandardScaling():
    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        return self
    def transform(self,X):
        return (X-self.mean_)/self.std_
class SoftMaxRegression():
    def __init__(self, random_state, epoch):
        self.random_state = random_state 
        self.epoch = epoch
        self.iteration = 0
        self.theta_ = None
    def partial_fit(self,X, y):
        assert self.iteration < self.epoch
        self.n_features_in_ = X.shape[1]
        self.labels_in_ = np.unique(y)
        if self.theta_ is None:
            np.random.seed(self.random_state)
            self.theta_ = np.zeros((len(self.labels_in_), self.n_features_in_ + 1, 1)) * 0.01
        X_b = add_dummy_feature(X)
        total_data = len(X)
        for iter in range(total_data):
            random_val = np.random.randint(total_data)
            Xi_b = X_b[random_val : random_val + 1] 
            yi = y[random_val : random_val + 1]
            eta = learning_rate(iter + self.iteration * total_data)
            temp_gradient = np.random.rand(len(self.labels_in_), self.n_features_in_ + 1, 1)
            for index in range(len(self.theta_)):
                y_val = int(yi[0] == self.labels_in_[index]) 
                gradients = 1 * Xi_b.T * (self.log_odds(self.theta_ , Xi_b , index) - y_val) 
                temp_gradient[index] = gradients
            self.theta_ = self.theta_ - eta * temp_gradient 
        self.iteration +=1
        return self

    def log_odds(self, theta , Xi_b , index):
        score = np.exp(Xi_b @ theta.squeeze(-1).T)
        numerator = score[0][index]
        denominator = score.sum()
        return numerator / denominator
    
    def predict_proba(self,theta, X):
        X = add_dummy_feature(X)
        theta =theta.squeeze(-1)
        self.predicted_value_ = np.zeros((len(X), len(theta[0])))
        for index, item in enumerate(X):
            score = np.exp(item @ theta.T)
            denominator = score.sum()
            self.predicted_value_[index] = score/denominator
        return self.predicted_value_

    def predict(self, theta, X):
        y_predicted = self.predict_proba(theta,X)
        return y_predicted.argmax(axis=1)
    
        

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


 

def main():
    iris = load_iris(as_frame = True)
    X = iris.data[["petal length (cm)", "petal width (cm)"]].values
    y = iris['target'].values
    train_test_split = TrainTestSplit(X,y,0.8,42)
    X_train , X_test , y_train , y_test , random_val= train_test_split.do_splitting()
    std_scale = StandardScaling()
    std_scale.fit(X_train)
    X_train = std_scale.transform(X_train)
    soft_regression = SoftMaxRegression(42 , 1000)  
    for _ in range(1000):
        soft_regression.partial_fit(X_train,y_train)
    X_test = std_scale.transform(X_test)
    y_pred = soft_regression.predict(soft_regression.theta_ , X_test)
    acc = accuracy(y_test , y_pred)
    print(acc)

    ########################### Plotting ##############################

    custom_cmap = ListedColormap(["#fafab0", "#9898ff", "#a0faa0"])

    x0 , x1 = np.meshgrid(
            np.linspace(0,8,500).reshape(-1,1),
            np.linspace(0,3.5, 200).reshape(-1,1),
            )
    X_new = np.c_[x0.ravel(), x1.ravel()]
    X_new = std_scale.transform(X_new)
    y_proba = soft_regression.predict_proba(soft_regression.theta_ , X_new)
    y_predict = soft_regression.predict(soft_regression.theta_, X_new)
    zz1 = y_proba[:, 1].reshape(x0.shape)
    zz = y_predict.reshape(x0.shape)
    plt.figure(figsize=(10, 4))
    plt.plot(X[y == 2, 0], X[y == 2, 1], "g^", label="Iris virginica")
    plt.plot(X[y == 1, 0], X[y == 1, 1], "bs", label="Iris versicolor")
    plt.plot(X[y == 0, 0], X[y == 0, 1], "yo", label="Iris setosa")

    plt.contourf(x0, x1, zz, cmap=custom_cmap)
    contour = plt.contour(x0, x1, zz1, cmap="hot")
    plt.clabel(contour, inline=1)
    plt.xlabel("Petal length")
    plt.ylabel("Petal width")
    plt.legend(loc="center left")
    plt.axis([0.5, 7, 0, 3.5])
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()
