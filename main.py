############################################################################
# LOGISTIC REGRESSION                                                      #
# Note: NJUST Machine Learning Assignment.                                 #
# Task: Binary Classification, Multi-class Classification.                 #
# Optimization: Grediant Descent (GD), Stochastic Grediant Descent(SGD).   #
# Author: Edmund Sowah                                                     #
############################################################################
import numpy as np
import matplotlib.pyplot as plt


def main():
    # Import and Read Data
    examX, examY = readData("data/exam_x.dat", "data/exam_y.dat")
    irisX, irisY = readData("data/iris_x.dat", "data/iris_y.dat")

    # Options Menus
    option = int(input("Please choose:\n"
                       "1. Exam dataset with Logistic-GD\n"
                       "2. Exam dataset with Logistic-SGD\n"
                       "3. Exam dataset with Softmax-GD\n"
                       "4. Exam dataset with Softmax-SGD\n"
                       "5. Iris dataset with Softmax-GD\n"
                       "6. Iris dataset with Softmax-SGD\n"))
    operations = {
        1: lambda: logisticRegression(examX, examY, alpha=0.01, mode="GD"),
        2: lambda: logisticRegression(examX, examY, alpha=0.03, mode="SGD"),
        3: lambda: softmaxRegression(examX, examY, alpha=0.01, mode="GD"),
        4: lambda: softmaxRegression(examX, examY, alpha=0.03, mode="SGD"),
        5: lambda: softmaxRegression(irisX, irisY, alpha=0.03, mode="GD"),
        6: lambda: softmaxRegression(irisX, irisY, alpha=0.2, mode="SGD")
    }

    # Plot Graph
    plt.ion()
    plt.figure(figsize=(10, 5))

    # Perform Regression
    operations[option]()

    # Plot Graph
    plt.ioff()
    plt.show()


def logisticRegression(X, Y, alpha, mode="GD"):
    # Function: Logistic regression
    theta = np.ones(3)
    costList = []
    for i in range(2000):
        leftPlt, rightPlt = setPlt(
            "Logistic Regression " + mode + " LR=" + str(alpha), X)
        # Plot data points
        leftPlt.scatter(X[:, 1], X[:, 2], c=Y)
        # Calculate the gradient
        if mode == "GD":  # Gradient descent
            gradient = np.sum(
                np.tile(Y - np.array([logiH(theta, x) for x in X]), (3, 1)).T * X, axis=0)
        else:  # Stochastic gradient descent
            r = np.random.randint(0, len(X))
            gradient = (Y[r] - logiH(theta, X[r])) * X[r]
        # Recompute theta
        theta += alpha * gradient
        # Calculate the cost
        cost = -np.sum(Y * [np.log(logiH(theta, x)) for x in X] +
                       (np.ones(len(Y)) - Y) * [np.log(1 - logiH(theta, x)) for x in X])
        costList.append(cost)
        print("cost: {}".format(cost))  # Temporary debug
        # Draw the decision plane
        x = np.linspace(np.min(X[:, 1] - X[:, 1].max() - X[:, 1].min() * 0.1),
                        np.max(X[:, 1]) + X[:, 1].max() - X[:, 1].min() * 0.1, 50)
        y = np.array([-(theta[1] * i + theta[0]) / theta[2] for i in x])
        leftPlt.plot(x, y, color=colors[i % len(colors)])
        # Drawing cost
        rightPlt.plot(range(1, len(costList) + 1), costList)
        # Drawing pause
        plt.pause(0.05 if mode == "GD" else 1e-6)


def softmaxRegression(X, Y, alpha, mode="GD"):
    # Function: Softmax regression
    C = len(set(Y))  # Number of categories
    Theta = np.ones((C, 3))
    costList = []
    for i in range(2000):
        leftPlt, rightPlt = setPlt(
            "Softmax Regression " + mode + " LR=" + str(alpha), X)
        # Plot data points
        leftPlt.scatter(X[:, 1], X[:, 2], c=Y)
        # Calculate the gradient
        if mode == "GD":  # Gradient descent
            Gradient = np.sum(np.tile(np.array([np.array(Y == k, dtype=int) for k in range(
                C)]).T - np.array([softH(Theta, x) for x in X]), (3, 1, 1)).T * np.tile(X, (C, 1, 1)), axis=1)
        else:  # Stochastic gradient descent
            r = np.random.randint(0, len(X))
            Gradient = np.tile(np.array([Y[r] == k for k in range(
                C)]) - softH(Theta, X[r]), (3, 1)).T * np.tile(X[r], (C, 1))
        # Recompute Theta
        Theta += alpha * Gradient
        # Calculate the cost
        cost = -np.array([np.log(softH(Theta, x)[y])
                         for x, y in zip(X, Y)]).sum()
        costList.append(cost)
        print("cost: {}".format(cost))
        # Draw the decision plane
        x = np.linspace(np.min(X[:, 1] - X[:, 1].max() - X[:, 1].min() * 0.1),
                        np.max(X[:, 1]) + X[:, 1].max() - X[:, 1].min() * 0.1, 800)
        for r, s, t in {2: [(0, 1, -1)], 3: [(0, 1, 2), (0, 2, 1), (1, 2, 0)]}[C]:
            y = np.array([((Theta[s][1] - Theta[r][1]) * i + Theta[s][0] - Theta[r][0]) / (
                Theta[r][2] - Theta[s][2] if not Theta[r][2] - Theta[s][2] == 0 else 1e-20) for i in x])
            index = (Theta[r][2] - Theta[t][2]) * y > (Theta[t][1] - Theta[r]
                                                       [1]) * x + Theta[t][0] - Theta[r][0] if not t == -1 else None
            leftPlt.plot(x[index] if not t == -1 else x, y[index]
                         if not t == -1 else y, color=colors[i % len(colors)])
        # Drawing cost
        rightPlt.plot(range(1, len(costList) + 1), costList)
        # Drawing pause
        plt.pause(0.05 if mode == "GD" else 1e-6)


def readData(xfile, yfile):
    # Function: read data set
    X = np.loadtxt(xfile)
    X = zScoreStd(X)  # z-score standardization
    X = np.column_stack((np.ones(len(X)), X))  # Add constant term
    Y = np.loadtxt(yfile, dtype=int)
    return X, Y


def zScoreStd(X):
    # Function: z-score standardization
    return (X - X.mean(axis=0)) / X.std(axis=0)


def logiH(theta, x):
    # Function: Logistic regression hypothesis equation
    return 1 / (1 + np.exp(-theta @ x))


def softH(Theta, x):
    # Function: softmax regression hypothetical equation
    return np.exp(Theta @ x) / (np.exp(Theta @ x)).sum()


def setPlt(title, X):
    # Function: Setting Plots
    plt.clf()
    plt.suptitle(title)
    leftPlt = plt.subplot(121)
    rightPlt = plt.subplot(122)
    leftPlt.set_xlabel("Feature 1")
    leftPlt.set_ylabel("Feature 2")
    rightPlt.set_xlabel("Iterations")
    rightPlt.set_ylabel("Cost")
    xRange = X[:, 1].max() - X[:, 1].min()
    yRange = X[:, 2].max() - X[:, 2].min()
    leftPlt.axis([X[:, 1].min() - xRange * 0.1, X[:, 1].max() + xRange *
                 0.1, X[:, 2].min() - yRange * 0.1, X[:, 2].max() + yRange * 0.1])
    return leftPlt, rightPlt


if __name__ == "__main__":
    colors = ("green", "yellow", "magenta", "cyan")  # Decision plane color
    main()
