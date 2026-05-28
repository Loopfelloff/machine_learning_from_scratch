#This contains my rbf seapration stuff
from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles

def polynomial_kernel(X1 , X2):
    return (np.dot(X1.squeeze() , X2.squeeze()) + 100) ** 10 

def predict_val(X_pred , X_train , y , alpha):
    prediction_list = []
    for each_x in X_pred:
        total_val = 0
        for index , item in enumerate(alpha):
            total_val += item * y[index] * polynomial_kernel(each_x , X_train[index]) 
        prediction_list.append(total_val)
    return np.array(prediction_list)

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

class SMO():
    def __init__(self, X , y , alpha, c , intercept):
        self.X_ = X
        self.alpha_ = alpha
        self.y_ = y
        self.c_ = c
        self.eps_ = 1e-05
        self.tol_ = 0.001
        self.intercept_ = intercept
        self.bound_ = list(range(len(self.X_)))
        self.unbound_ = []

    def takeStep(self, index_one , index_two , E2):

        if(index_one == index_two):
            return False

        alpha_one = self.alpha_[index_one]
        alpha_two = self.alpha_[index_two]
        y1 = self.y_[index_one]
        y2 = self.y_[index_two]
        x1 = self.X_[index_one]
        x2 = self.X_[index_two]
        
        predicted_for_one = predict_val(x1.reshape((1,2)) , self.X_ , self.y_ , self.alpha_ )
        E1 = predicted_for_one[0] - y1 # fix
        s = y1 * y2
        if (y1 == y2):
            L = max(0 , alpha_two + alpha_one - self.c_)
            H = min(self.c_ , alpha_one + alpha_two)
        else:
            L = max(0, alpha_two - alpha_one)
            H = min(self.c_, self.c_ + alpha_two - alpha_one)

        if L == H:
            return False


        k11 = polynomial_kernel(x1 , x1)
        k12 = polynomial_kernel(x1 , x2)
        k22 = polynomial_kernel(x2 , x2)

        eta = k11 + k22 - 2 * k12

        if (eta > 0):
            a2 = alpha_two  + y2 * (E1 - E2) / eta
            if (a2 < L):
                a2 = L
            elif(a2 > H):
                a2 = H
        
        else :
            f1 = y1 * (E1 + self.intercept_[0][0]) - alpha_one * k11 - s * alpha_two * k12 
            f2 = y2 * (E2 + self.intercept_[0][0]) - s * alpha_one * k12 - alpha_two * k22 
            L1 = alpha_one + s * (alpha_two - L)
            H1 = alpha_one + s * (alpha_two - H )
            Lobj = L1 * f1 + L * f2 + (1/2) * (L1 ** 2) * k11 + (1/2) * (L ** 2) * k22 + s * L * L1 * k12 
            Hobj = H1 * f1 + H * f2 + (1/2) * (H1 ** 2) * k11 + (1/2) * (H ** 2) * k22 + s * H * H1 * k12 
         
            if (Lobj < Hobj-self.eps_):
                a2 = L
            elif (Lobj >  Hobj+self.eps_):
                a2 = H
            else:
                a2 = alpha_two

        if (np.abs(a2 - alpha_two) < self.eps_*(a2 +alpha_two + self.eps_)):
            return False
        a1 = alpha_one + s * (alpha_two - a2)

        b1 = E1 + y1 * (a1 - alpha_one) * k11 + y2 * (a2 - alpha_two) * k12 + self.intercept_[0][0]
        b2 = E2 + y1 * (a1 - alpha_one) * k12 + y2 * (a2 - alpha_two) * k22 + self.intercept_[0][0]

        if ((a1 >0 and a1 < self.c_) and (a2 <= 0 or a2 >= self.c_ )):
            self.intercept_ = np.float32(b1).reshape((1,1))
        elif ((a1 <= 0 or a1 >= self.c_) and (a2 > 0 and a2 < self.c_ )):
            self.intercept_ = np.float32(b2).reshape((1,1))
        else:
            self.intercept_ = np.float32((b1+b2)/2).reshape((1,1))

        self.alpha_[index_one] = a1
        self.alpha_[index_two] = a2

        return True

    def set_bound_unbound(self):
        temp_bound = []
        temp_unbound = []

        for index in range(len(self.alpha_)):

            if 0 < self.alpha_[index] < self.c_:
                temp_unbound.append(index)
            else:
                temp_bound.append(index)
        
        self.bound_ = temp_bound.copy()
        self.unbound_ = temp_unbound.copy() # i am just being cautious

    def max_absolute_error(self , E2):
        to_compare_y  = self.y_[self.unbound_]
        to_compare_x = self.X_[self.unbound_]
        predicted_for_one = predict_val(to_compare_x , self.X_ , self.y_ , self.alpha_ )
        full_err = predicted_for_one - to_compare_y # fix 
        abs_err = np.abs(full_err - E2)

        return self.unbound_[abs_err.argmax()] 

    def examineExample(self, index_two):
        y2 = self.y_[index_two]
        alpha_two  = self.alpha_[index_two]
        predicted_for_one = predict_val(self.X_[index_two].reshape((1,2)) , self.X_ , self.y_ , self.alpha_ )
        E2 = predicted_for_one[0] - y2 
        r2 = E2 * y2
        if len(self.bound_) > 0:
            random_bound_index = np.random.randint(len(self.bound_))
        else:
            random_bound_index = 0
        if len(self.unbound_) > 0:
            random_unbound_index = np.random.randint(len(self.unbound_))
        else:
            random_unbound_index = 0
        if ((r2 < - self.tol_ and alpha_two < self.c_) or (r2 > self.tol_ and alpha_two > 0)):
            if(len(self.unbound_) > 1):
                index_one = self.max_absolute_error(E2)
                if self.takeStep(index_one , index_two , E2) == True:
                    self.set_bound_unbound()
                    return True
            for index in range(len(self.unbound_)):
                index_one = self.unbound_[(index + random_unbound_index) % len(self.unbound_)]
                if self.takeStep(index_one, index_two , E2) == True:
                    self.set_bound_unbound()
                    return True

            for index in range(len(self.bound_)):
                index_one = self.bound_[(index + random_bound_index) % len(self.bound_)]
                if self.takeStep(index_one , index_two , E2) == True:
                    self.set_bound_unbound()
                    return True

        return False

    def main_routine(self):
        numChanged = 0
        examineAll = True
        iter = 0
        while(numChanged > 0 or examineAll == True):
            numChanged = 0
            if(examineAll == True):
                for index in range(len(self.X_)):
                    iter +=1
                    numChanged += self.examineExample(index)
            else:
                for index in self.unbound_:
                    iter +=1
                    numChanged += self.examineExample(index)
            if(examineAll == True):
                examineAll = False
            elif(numChanged == 0):
                examineAll = True
        print(iter)
        return self.intercept_ ,self.alpha_
        

class SVM():
    def __init__(self , C):
        self.c_ = C

    def fit(self , X , y):
        np.random.seed(42)
        self.X = X
        self.intercept_ = np.random.rand(1,1)
        self.y_b = y.astype("int8").copy()
        self.y_b[self.y_b == 0] = -1
        total_data = len(self.X)
        self.alpha_ = np.zeros((total_data , )) 
        smo = SMO(self.X , self.y_b , self.alpha_.copy() , self.c_, self.intercept_.copy())
        self.intercept_ ,self.alpha_= smo.main_routine()
        return self
    
    def predict(self, X):
        self.predict_proba_ = predict_val(X , self.X , self.y_b , self.alpha_ )
        return self.predict_proba_ >= 0



def main():
    X ,y = make_circles(n_samples=500 , factor=0.5 , noise=0.05 , random_state=42)
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
    print(svm.intercept_)

    ############### Plotting Purposes #########################

    custom_cmap = ListedColormap(["#fafab0", "#9898ff", "#a0faa0"])

    x0 , x1 = np.meshgrid(
            np.linspace(-1,1,500).reshape(-1,1),
            np.linspace(-1,1, 200).reshape(-1,1),
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
    plt.axis([-1, 1, -1, 1])
    plt.grid()
    ##################################################
    plt.plot(X[ y == 0 , 0 ],  X[y == 0 , 1] , "o")
    plt.plot(X[ y == 1 , 0 ],  X[y == 1 , 1] , "ro")
    plt.xlabel("petal length (cm)")
    plt.ylabel("petal width (cm)")
    
    plt.show()
main()





