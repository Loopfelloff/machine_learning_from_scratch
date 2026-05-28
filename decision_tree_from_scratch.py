import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

#values , count = np.unique(check_arr , return_counts=True)
class DecisionTreeNode():
    def __init__(self):
        self.left = None 
        self.right = None
        self.gini = 0 
        self.threshold_class = 0
        self.threshold = 0
        self.samples = 0
        self.value = np.array([])
        self.class_name = 0

def return_gini_impurity(count):
    return 1- np.sum( (count / np.sum(count))**2 )

def cost_function(m_left,  m , G_left, G_right):
    return (m_left/m) * G_left + ((m-m_left)/m) * G_right

def find_optimal(X_original , y_original , root_node : DecisionTreeNode,X_first , y_first , X_second , y_second):
    min_j_val_first = np.inf
    min_j_val_second = np.inf
    min_first_tk = np.inf 
    min_second_tk = np.inf
    threshold_feat = 0
    total_min_j_val = np.inf
    total_min_threshold = np.inf
    for index in range(len(X_first)-1):
        if X_first[index]  == X_first[index+1]:
            continue
        k = index
        t_k = (X_first[k] + X_first[k+1])/2
        _, first_count_left = np.unique(y_first[:index+1] , return_counts=True) 
        _, first_count_right = np.unique(y_first[index+1:] , return_counts=True) 
        g_left = return_gini_impurity(first_count_left)
        g_right = return_gini_impurity(first_count_right)
        m_left = index + 1
        j_val = cost_function(m_left , len(X_first) , g_left , g_right)
        if j_val < min_j_val_first:
            min_j_val_first = j_val
            min_first_tk = t_k

    for index in range(len(X_second)-1):
        if X_second[index]  == X_second[index+1]:
            continue
        k = index
        t_k = (X_second[k] + X_second[k+1])/2
        _, second_count_left = np.unique(y_second[:index+1] , return_counts=True) 
        _, second_count_right = np.unique(y_second[index+1:] , return_counts=True) 
        g_left = return_gini_impurity(second_count_left)
        g_right = return_gini_impurity(second_count_right)
        m_left = index + 1
        j_val = cost_function(m_left , len(X_second) , g_left , g_right)
        if j_val < min_j_val_second:
            min_j_val_second = j_val
            min_second_tk = t_k
    
    if min_j_val_second <= min_j_val_first:
        total_min_j_val = min_j_val_second
        total_min_threshold = min_second_tk
        threshold_feat = 1
    else: 
        total_min_j_val = min_j_val_first
        total_min_threshold = min_first_tk
        threshold_feat = 0

    vals , count = np.unique(y_first , return_counts=True) 
    gini_impurity_root = return_gini_impurity(count) 
    root_node.gini = gini_impurity_root
    root_node.samples = len(y_first)
    root_node.value = count
    root_node.class_name = int(vals[np.argmax(count)])
    if gini_impurity_root <= total_min_j_val:
        return
    else:
        root_node.threshold = total_min_threshold.round(2)
        root_node.threshold_class = threshold_feat
        left_node = DecisionTreeNode()
        right_node = DecisionTreeNode()
        root_node.left = left_node
        root_node.right = right_node 
        if threshold_feat == 0:
            left_mask = X_original[: , 0] <= total_min_threshold
        else:
            left_mask = X_original[: , 1] <= total_min_threshold

        X_orig_left, y_orig_left = X_original[left_mask], y_original[left_mask]

        X_orig_right, y_orig_right = X_original[~left_mask], y_original[~left_mask]

        idx_first_left = np.argsort(X_orig_left[:, 0])
        idx_second_left = np.argsort(X_orig_left[:, 1])

        idx_first_right = np.argsort(X_orig_right[:, 0])
        idx_second_right = np.argsort(X_orig_right[:, 1])

        find_optimal(
                     X_orig_left, y_orig_left,left_node,
                     X_orig_left[idx_first_left, 0], y_orig_left[idx_first_left],
                     X_orig_left[idx_second_left, 1], y_orig_left[idx_second_left]
)

        find_optimal(
                     X_orig_right, y_orig_right,right_node,
                     X_orig_right[idx_first_right, 0], y_orig_right[idx_first_right],
                     X_orig_right[idx_second_right, 1], y_orig_right[idx_second_right]
                     )



def create_tree(X_original , y_original , X_first , y_first , X_second , y_second):
    root_node = DecisionTreeNode()
    find_optimal(X_original, y_original,root_node , X_first , y_first , X_second , y_second) 
    return root_node

def find_class(root_node : DecisionTreeNode , X):
    
    next_node = root_node
    selected_class = None
    while next_node is not None:
        threshold = next_node.threshold 
        selected_class = next_node.class_name
        if next_node.left is None and next_node.right is None:
            break  
        threshold_class = next_node.threshold_class
        if threshold_class == 0:
            if X[0] <= threshold:
                next_node = next_node.left
            else:
                next_node = next_node.right
        else:
            if X[1] <= threshold:
                next_node = next_node.left
            else:
                next_node = next_node.right
    return selected_class

class DecisionTreeClassifier():
    def fit(self , X , y):
        self.X_ = X
        self.y_ = y
        idx_first_feat = np.argsort(self.X_[: , 0] , axis=0)
        self.X_first_sorted_ = self.X_[: , 0][idx_first_feat].copy()
        self.y_first_sorted_ = self.y_[idx_first_feat].copy()
        idx_second_feat = np.argsort(self.X_[: , 1] , axis=0)
        idx_second_feat = idx_second_feat.reshape(-1)
        self.X_second_sorted_ = self.X_[: , 1][idx_second_feat].copy()
        self.y_second_sorted_ = self.y_[idx_second_feat].copy()
        self.root_node_ = create_tree(self.X_, self.y_ , self.X_first_sorted_ , self.y_first_sorted_, self.X_second_sorted_, self.y_second_sorted_) 

    def predict(self , X):
        predicted_val = []
        for item in X:
            predicted_val.append(find_class(self.root_node_, item))
        return predicted_val
def accuracy(y_predict, y_test):
    
    values , count = np.unique(np.array(y_predict) == np.array(y_test) , return_counts=True) 
    
    return values, count

def main():
    iris = load_iris(as_frame=True)
    X_iris = iris.data[["petal length (cm)" , "petal width (cm)"]].values
    y_iris = iris.target.values
    decision_clf = DecisionTreeClassifier()
    decision_clf.fit(X_iris , y_iris)
    y_predict = decision_clf.predict(X_iris)
        ############### Plotting Purposes #########################

    custom_cmap = ListedColormap(["#fafab0", "#9898ff", "#a0faa0"])

    x0 , x1 = np.meshgrid(
            np.linspace(-1,7,500).reshape(-1,1),
            np.linspace(-1,3, 200).reshape(-1,1),
            )
    X_new = np.c_[x0.ravel(), x1.ravel()]
    y_predict_plot = np.array(decision_clf.predict(X_new)) 
    zz = y_predict_plot.reshape(x0.shape)
    plt.figure(figsize=(10, 4))
    plt.contourf(x0, x1, zz, cmap=custom_cmap)
    plt.xlabel("Petal length")
    plt.ylabel("Petal width")
    plt.grid()
    ##################################################
    plt.plot(X_iris[ y_iris == 0 , 0 ],  X_iris[y_iris == 0 , 1] , "o")
    plt.plot(X_iris[ y_iris == 1 , 0 ],  X_iris[y_iris == 1 , 1] , "ro")
    plt.plot(X_iris[ y_iris == 2 , 0 ],  X_iris[y_iris == 2 , 1] , "go")
    plt.xlabel("petal length (cm)")
    plt.ylabel("petal width (cm)")
    
    plt.show()
 
if __name__ == "__main__":
    main()


