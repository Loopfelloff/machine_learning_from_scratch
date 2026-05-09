import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

epoches=1000
m = 100
t0, t1 = 5 ,50

def learning_schedule(t):
    return t0/(t + t1)


def main():
    y_val = []
    x_val = []
    for epoch in range(epoches):
        for iteration in range(m):
            eta = learning_schedule(epoch*m + iteration)
            x_val.append(epoch*m + iteration)
            y_val.append(eta)
    plt.plot(x_val, y_val , 'b')
    plt.show()
if __name__ == "__main__":
    main()
