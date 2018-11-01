from __future__ import division
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
# from scipy.integrate import RK45

def log_mag(x):
    norm2 = np.linalg.norm(x)
    return np.log(norm2**0.5)

def q1():
    dxdt = lambda x, y, z: 10 * (y - x)
    dydt = lambda x, y, z: (28 * x) - y - (x * z)
    dzdt = lambda x, y, z: (x * y) - (8 / 3) * z

    def RK4_for_3var(e, f, g, init, interval, h):
        hby2 = h / 2
        a, b = interval
        fns = [e, f, g]

        tk = a  # initialise t
        xk, yk, zk = init  # initialise x, y, z
        
        xs, ys, zs = [xk], [yk], [zk]
        
        while tk <= b:

            f1s = [fn(xk, yk, zk) for fn in fns]
            f2s = [fn(xk + hby2*f1s[0], yk + hby2*f1s[1], zk + hby2*f1s[2]) for fn in fns]
            f3s = [fn(xk + hby2*f2s[0], yk + hby2*f2s[1], zk + hby2*f1s[2]) for fn in fns]
            f4s = [fn(xk + h*f3s[0], yk + h*f3s[1], zk + h*f3s[2]) for fn in fns]
            
            xk += (h / 6) * (f1s[0] + (2 * f2s[0]) + (2 * f3s[0]) + f4s[0]); xs.append(xk)
            yk += (h / 6) * (f1s[1] + (2 * f2s[1]) + (2 * f3s[1]) + f4s[1]); ys.append(yk)
            zk += (h / 6) * (f1s[2] + (2 * f2s[2]) + (2 * f3s[2]) + f4s[2]); zs.append(zk)
            
            tk += h
        
        return xs, ys, zs

    TOLERANCE = 10e-6
    INIT = [3, 3, 20]
    INTERVAL = (0, 100)

    converged = False

    # while not converged:
    #     RK4(dxdt, dydt, dzdt, INIT, INTERVAL, h)
    #     converged = error < TOLERANCE

    xs, ys, zs = RK4_for_3var(dxdt, dydt, dzdt, INIT, INTERVAL, 0.001)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(xs, ys, zs, marker='.', color='red')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    plt.show()



def q2():

    def euler(f, init, interval, h):
        a, b = interval

        tk = a  # initialise t
        xk = init  # initialise x, y, z
        
        xs = [xk]
        errors = []

        while tk <= b:
            
            del_x = h * f(tk, xk)
            xk += del_x
            
            xs.append(xk)
            errors.append(abs(np.cos(tk) - xk))
            
            tk += h
        
        return xs, errors

    def RK4(f, init, interval, h):
        hby2 = h / 2
        a, b = interval

        tk = a  # initialise t
        xk = init  # initialise x, y, z
        
        xs = [xk]
        errors = []
        
        while tk <= b:

            f1 = f(tk, xk)
            f2 = f(tk + hby2, xk + hby2*f1)
            f3 = f(tk + hby2, xk + hby2 * f2)
            f4 = f(tk + h, xk + h*f3)
            
            del_x = (h / 6) * (f1 + (2 * f2) + (2 * f3) + f4)
            xk += del_x
            
            xs.append(xk)
            errors.append(abs(np.cos(tk) - xk))
            
            tk += h
        
        return xs, errors

    # def ode45(f):
    #     return scipy.integrate.solve_ivp(f)

    dydt = lambda t, y: np.sin(t)*(y**2 - np.cos(t)**2 - 1)

    hs = []
    norm_errors = []

    h = 1
    while h > 0.001:
        ys, errors = RK4(dydt, 1, (0, 50), h)
        hs.append(np.log(h))
        norm_errors.append(log_mag(errors))
        h /= 2
    
    plt.plot(hs, norm_errors)
    plt.title("loglog of error vs h")
    plt.xlabel("log h")
    plt.ylabel("log norm of error")
    plt.show()

    hs = []
    norm_errors = []
    
    h = 1
    while h > 0.001:
        ys, errors = euler(dydt, 1, (0, 50), h)
        hs.append(np.log(h))
        norm_errors.append(log_mag(errors))
        h /= 2

    plt.plot(hs, norm_errors, color="red")
    plt.title("loglog of error vs h")
    plt.xlabel("log h")
    plt.ylabel("log norm of error")
    plt.show()

    ys, errors = RK4(dydt, 1, (0, 500), 0.1)
    true_plot = [np.cos(t) for t in np.arange(0, 500, 0.1)]
    plt.plot(range(len(ys)), ys, color="red"); plt.plot(range(len(true_plot)), true_plot, color="blue"); 
    plt.title("cos(t) vs RK4")
    plt.xlabel("t")
    plt.ylabel("value")
    plt.show()


def q3():
    dxdt = lambda x, y: -4*y + x*(1 - x**2 - y**2)
    dydt = lambda x, y: 4*x + y*(1 - x**2 - y**2)

    INIT = [0.9, 0.5]
    INIT2 = [2, 3]

    INIT = [[0.5*np.cos(t), 0.5*np.sin(t)] for t in np.arange(0,10,0.5)]
    INIT2 = [[3*np.cos(t), 3*np.sin(t)] for t in np.arange(0,10,0.5)]

    INTERVAL = (0, 10)

    def RK4_for_2var(f, g, init, interval, h):
        hby2 = h / 2
        a, b = interval
        fns = [f, g]

        tk = a  # initialise t
        xk, yk = init  # initialise x, y, z
        
        xs, ys = [xk], [yk]
        
        while tk <= b:

            f1s = [fn(xk, yk) for fn in fns]
            f2s = [fn(xk + hby2*f1s[0], yk + hby2*f1s[1]) for fn in fns]
            f3s = [fn(xk + hby2*f2s[0], yk + hby2*f2s[1]) for fn in fns]
            f4s = [fn(xk + h*f3s[0], yk + h*f3s[1]) for fn in fns]
            
            xk += (h / 6) * (f1s[0] + (2 * f2s[0]) + f4s[0]); xs.append(xk)
            yk += (h / 6) * (f1s[1] + (2 * f2s[1]) + f4s[1]); ys.append(yk)
            
            tk += h
        
        return xs, ys

    for i in INIT:
        xs, ys = RK4_for_2var(dxdt, dydt, i, INTERVAL, 0.01)
        plt.plot(xs, ys)
    plt.title("Init within circle")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

    for i in INIT2:
        xs1, ys1 = RK4_for_2var(dxdt, dydt, i, INTERVAL, 0.01)
        plt.plot(xs1, ys1)
    plt.title("Init outside circle")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


# def ode45(f):
#     t = np.arange(0, 100)

#     return odeint(pend, f, t, args=(0, 100))

# dydt = lambda t, y: np.sin(t)*(y**2 - np.cos(t)**2 - 1)
# ode45(dydt)

q2()