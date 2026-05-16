# this is where i test circles
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt

# 1. Generate concentric circles
X, y = make_circles(n_samples=500, factor=0.5, noise=0.05, random_state=42)
print(X[:5])
print(y[:5])

# 2. Plot the dataset
plt.scatter(X[y == 0, 0], X[y == 0, 1], color='red', label='Outer Circle')
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', label='Inner Circle')
plt.legend()
plt.show()

