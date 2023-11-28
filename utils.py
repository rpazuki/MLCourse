import numpy as np
import matplotlib.pyplot as plt

def plot_Hill_general(ax):
    ax.set_xlabel("concentration", fontsize=18)
    ax.set_ylabel("Fluorescent intensity", fontsize=18)
    ax.grid()
    ax.legend(prop={'size': 12})
    
def plot_ROC_general(ax, title):
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    plt.title(title)
    ax.legend(loc="lower right", prop={'size': 12})
    ax.grid()
    
    
def train_test(xs, ys, train_size):
    assert len(xs) == len(ys)
    data = np.vstack([xs, ys])
    indices = sorted(np.random.choice(range(data.shape[1]), train_size, replace=False)) 
    
    train = data[:, indices]
    test_indeces = [index for index in range(data.shape[1]) if index not in indices]
    test = data[:, test_indeces]
    return (train[0, :], train[1, :], test[0, :], test[1, :])

def classification_2D_dataset():
    class_1 = np.random.multivariate_normal([1.0,1.0],
                                        [[.5, 0],[0, 1.0]], 
                                        500)
    class_2 = np.random.multivariate_normal([4.5, 3.5],
                                            [[6, -2], [-2, 3.5]], 
                                            500)
    
    class_3 = np.random.multivariate_normal([11.0, 1.0],
                                            [[2, -1], [-1, 2]], 
                                            500)
    X = np.vstack([class_1, class_2, class_3])
    Y = np.concatenate([np.repeat(0, class_1.shape[0]),
                        np.repeat(1, class_2.shape[0]),
                        np.repeat(2, class_3.shape[0])])
    return X,Y