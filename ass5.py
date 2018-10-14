import numpy as np
import matplotlib.pyplot as plt

def f1(x):
    return x ** 2

def der_h4(x, h, f=f1):
    return (-f(x + 2 * h) + 8 * f(x + h) - 8 * f(x - h) + f(x - 2 * h)) / 12 * h
    
def der_h1(x, h, f=f1):
    forward = (f(x + h) - f(x)) / h
    backward = (f(x) - f(x - h)) / h
    return [forward, backward]

def q1():
    xs = np.arange(-5, 5, 0.1)
    h4s = []
    for h in [0.5, 0.25, 0.2, 0.1, 0.01]:
        h4, hf, hb = [], [], []
        
        for x in xs:
            h4.append(der_h4(x, h))
            h1 = der_h1(x, h)
            hf.append(h1[0])
            hb.append(h1[1])
        plt.plot(xs, h4); plt.plot(xs, hf); plt.plot(xs, hb)
        plt.title("comparision")
        plt.gca().legend(("h4", "h1 forward", "h1 backward"))
        plt.xlabel("x")
        plt.ylabel("derivative")
        plt.show()
        h4s.append(h4)
#     plt.plot(xs, h4[:100], h4[100:200], h4[200:300], h4[300:400], h4[400:500])
#     plt.title("different h")
#     plt.xlabel("x")
#     plt.ylabel("derivative")
#     plt.gca().legend(('0.1', '0.01', '0.001', '0.0001', '0.00001'))
#     plt.show()
        
def f2(x):
    return x**2

def trapezoid(a, b, h=0.1, f=f2):
    I = 0
    I += f(a) + f(b)
    for each in np.arange(a + h, b, h):
        I += 2 * f(each)
    return (h/2)*I

def simpson(a, b, h=0.5, f=f2):
    I = 0
    I += f(a) + f(b)
    for each in np.arange(a + 2*h, b, 2*h):
        I += 2 * f(each)
    for each in np.arange(a + h, b, 2*h):
        I += 4 * f(each)
    return (h/3)*I

def gauss_2(a, b, f=f2):
    c, d = b - a, b + a
    t = lambda x: -((x * c) - d) / 2
    x0 = (1/3)**0.5
    I = f(t(-x0)) + f(t(x0))
    return I*(c/2)

def gauss_3(a, b, f=f2):
    c, d = b - a, b + a
    t = lambda x: -((x * c) - d) / 2
    x0 = (3 / 5)** 0.5
    I = (5*f(t(-x0)) + 5*f(t(x0)) + 8*f(t(0)))/9
    return I * (c / 2)

def adaptive(a, b):
    i = 0
    converged = False
    prev = 0
    while not converged:
        h = (b - a) / (2**i)
        I = simpson(a, b, h)
        converged = abs(I - prev) < 0.0001
        prev = I
    return I

def q2():
    for fn in [trapezoid, simpson, gauss_2, gauss_3, adaptive]:
        print(fn(0, 2))

