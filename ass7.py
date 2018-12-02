import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import solve
from time import time

np.set_printoptions(3)
DEBUG = False

def q3():
    NUM_ROWS = 150 # for t
    NUM_COLS = 150 # for x

    c = 1
    h = 1.0 / NUM_COLS
    k = 1.0 / NUM_ROWS

    r = c * h / k

    r2 = r**2

    if r >= 1: print("Unstable r >= 1\n")

    u = np.zeros((NUM_ROWS, NUM_COLS))
    if DEBUG: print("Initialising the grid\n", u.T)

    # fixing initial conditions
    u[:-1,0] = [np.sin(np.pi * i) for i in np.arange(0, 1, h)][:NUM_COLS-1]
    if DEBUG: print("Applying u(x,0) = sin(x)\n", u.T)


    # finding the second row using taylor appr.
    u[:, 1] = u[:, 0]  # since du/dt = 0
    if DEBUG: print("Finding second row\n", u.T)

    # from 3rd row onwards
    for j in range(2, NUM_ROWS):
        for i in range(1,NUM_COLS-1):
            u[i,j] = 2*(1 - r2)*u[i,j-1] + r2*(u[i+1,j-1] + u[i-1,j-1]) - u[i,j-2]

    if DEBUG: print("After solving for all points\n", u.T)

    # 2d viz
    # plt.matshow(u)
    # plt.colorbar()
    # plt.show()

    # surface plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x = np.arange(0, NUM_COLS, 1)
    y = np.arange(0, NUM_ROWS, 1)
    plt.xlabel("time"); plt.ylabel("x"); plt.title("Heat equation solution")
    X, Y = np.meshgrid(x, y)
    surf = ax.plot_surface(X, Y, u, rstride=1, cstride=1, cmap=cm.coolwarm, antialiased=True)
    plt.show()

    # contour plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.contour(X, Y, u, cmap=cm.coolwarm, antialiased=True)
    plt.show()


def q2():
    NUM_ROWS = 25
    NUM_COLS = 300

    c = 1
    k = 0.1 / NUM_COLS # for t
    h = 1 / NUM_ROWS # for x

    r = (c ** 2) * k / (h ** 2)

    if r >= 0.5: print("Unstable r >= 0.5\n")

    u = np.zeros((NUM_ROWS, NUM_COLS))
    if DEBUG: print("Initialising the grid\n", u.T)

    u[1:-1, 0] = [np.sin(np.pi * i) + np.sin(2 * np.pi * i) for i in np.arange(0, 1, h)][1:-1]
    if DEBUG: print("Applying u(x,0) = sin(x) + sin(2x)\n", u.T)

    # from 2nd row onwards
    for j in range(1, NUM_COLS):
        for i in range(1,NUM_ROWS-1):
            u[i,j] = (1 - 2*r)*u[i,j-1] + r*(u[i+1,j-1] + u[i-1,j-1])

    if DEBUG: print("After solving for all points\n", u.T)

    # surface plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x = np.arange(0, NUM_COLS, 1)
    y = np.arange(0, NUM_ROWS, 1)
    plt.xlabel("time"); plt.ylabel("x"); plt.title("Heat equation solution")
    X, Y = np.meshgrid(x, y)
    surf = ax.plot_surface(X, Y, u, rstride=1, cstride=1, cmap=cm.coolwarm, antialiased=True)
    plt.show()

    # contour plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.contour(X, Y, u, cmap=cm.coolwarm, antialiased=True)
    plt.show()

def q2b():
    NUM_ROWS = 25
    NUM_COLS = 300

    c = 1
    k = 0.1 / NUM_COLS # for t
    h = 1 / NUM_ROWS # for x

    r = (c ** 2) * k / (h ** 2)

    if r >= 0.5: print("Unstable r >= 0.5\n")

    u = np.zeros((NUM_ROWS, NUM_COLS))
    if DEBUG: print("Initialising the grid\n", u.T)

    u[1:-1, 0] = [np.sin(np.pi * i) + np.sin(2 * np.pi * i) for i in np.arange(0, 1, h)][1:-1]
    if DEBUG: print("Applying u(x,0) = sin(x) + sin(2x)\n", u.T)


    tdm = np.zeros((NUM_ROWS - 2, NUM_ROWS - 2))

    # constructing tdm
    for i in range(tdm.shape[0]):
        tdm[i, i] = 2 + 2 * r
    for i in range(0, tdm.shape[0]-1):
        tdm[i + 1, i] = -r
        tdm[i, i + 1] = -r

    if DEBUG: print(tdm)

    for j in range(0, NUM_COLS-1):
        b = [u[3,j]] + [u[i, j] + u[i+2,j] for i in range(2, NUM_ROWS - 2)] + [u[NUM_ROWS - 2,j]]
        ans = solve(tdm, b)
        u[1:-1, j+1] = ans

    if DEBUG: print("After solving for all points\n", u.T)

    # surface plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x = np.arange(0, NUM_COLS, 1)
    y = np.arange(0, NUM_ROWS, 1)
    plt.xlabel("time"); plt.ylabel("x"); plt.title("Heat equation solution")
    X, Y = np.meshgrid(x, y)
    surf = ax.plot_surface(X, Y, u, rstride=1, cstride=1, cmap=cm.coolwarm, antialiased=True)
    plt.show()

    # contour plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.contour(X, Y, u, cmap=cm.coolwarm, antialiased=True)
    plt.show()

def q1(method=5, w=1.4):
    start = time()
    h = 0.025
    k = 0.025

    ITERATIONS = range(100)

    NUM_ROWS = int(1.0 / h)  # for x
    NUM_COLS = int(1.0 / k)  # for y

    # NUM_COLS = 4
    # NUM_ROWS = 4

    # x is colums
    # y is rows

    u = np.zeros((NUM_ROWS, NUM_COLS))

    # init conditions
    u[0, :] = 100
    u[:, 0] = 100

    if DEBUG: print("Applying 100\n", u.T)

    for _ in ITERATIONS:
        for i in range(1,NUM_ROWS-1):
            for j in range(1, NUM_COLS - 1):
                prev = u[i,j]
                if method == 5:
                    u[i, j] = (u[i - 1, j] + u[i, j + 1] + u[i + 1, j] + u[i, j - 1]) / 4
                else:
                    u[i, j] = (u[i - 1, j] + u[i, j + 1] + u[i + 1, j] + u[i, j - 1]
                            + u[i - 1, j + 1] + u[i + 1, j + 1] + u[i + 1, j - 1] + u[i - 1, j - 1]) / 8
                    
                u[i, j] = w * u[i, j] + (1 - w) * prev
        
        i = NUM_ROWS - 1
        for j in range(1, NUM_COLS - 1):
            prev = u[i,j]
            if method == 5:
                u[i, j] = (2*u[i - 1, j] + u[i, j + 1] + u[i, j - 1]) / 4
            else:
                u[i, j] = (2*u[i - 1, j] + u[i, j + 1] + u[i, j - 1]
                        + 2*u[i - 1, j + 1] + 2*u[i - 1, j - 1]) / 8
                
            u[i, j] = w * u[i, j] + (1 - w) * prev
        
        j = NUM_COLS - 1
        for i in range(1, NUM_ROWS - 1):
            prev = u[i,j]
            if method == 5:
                u[i, j] = (u[i - 1, j] + u[i + 1, j] + 2*u[i, j - 1]) / 4
            else:
                u[i, j] = (u[i - 1, j] + u[i + 1, j] + 2*u[i, j - 1]
                            + 2*u[i + 1, j - 1] + 2*u[i - 1, j - 1]) / 8
                
            u[i, j] = w * u[i, j] + (1 - w) * prev
        
        i, j = NUM_ROWS - 1, NUM_COLS - 1
        if method == 5:
            u[i, j] = (2 * u[i - 1, j] +  2 * u[i, j - 1]) / 4
        else:
            u[i, j] = (2 * u[i - 1, j] + 2 * u[i, j - 1] + 4 * u[i - 1, j - 1]) / 8
        
        if DEBUG: print("After solving for all points\n", u.T)

    # surface plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x = np.arange(0, NUM_COLS, 1)
    y = np.arange(0, NUM_ROWS, 1)
    plt.xlabel("time"); plt.ylabel("x"); plt.title("Heat equation solution")
    X, Y = np.meshgrid(x, y)
    surf = ax.plot_surface(X, Y, u, rstride=1, cstride=1, cmap=cm.coolwarm, antialiased=True)
    plt.show()

    # contour plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.contour(X, Y, u, cmap=cm.coolwarm, antialiased=True)
    plt.show()

    end = time()
    print("time taken : {}".format((end-start)*0.001))       

q1()