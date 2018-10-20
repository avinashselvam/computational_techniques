from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import time

def f1(x):
    return x ** 2

def der_h4(x, h, f=f1):
    return (-f(x + 2 * h) + 8 * f(x + h) - 8 * f(x - h) + f(x - 2 * h)) / (12 * h)
    
def der_h1(x, h, f=f1):
    forward = (f(x + h) - f(x)) / h
    backward = (f(x) - f(x - h)) / h
    return [forward, backward]

# def q1():
#     xs = np.arange(-5, 5, 0.1)
#     h4s = []
#     for h in [0.5, 0.25, 0.2, 0.1, 0.01]:
#         h4, hf, hb = [], [], []
        
#         for x in xs:
#             h4.append(der_h4(x, h))
#             h1 = der_h1(x, h)
#             hf.append(h1[0])
#             hb.append(h1[1])
        
#         # comparing h4, forward, backeard for different h
#         plt.plot(xs, h4); plt.plot(xs, hf); plt.plot(xs, hb)
#         plt.title("comparision")
#         plt.gca().legend(("h4", "h1 forward", "h1 backward"))
#         plt.xlabel("x"); plt.ylabel("derivative"); plt.show()

#         h4s.append(h4)

#     # comparing different h
#     for i in range(5): plt.plot(xs, h4s[i])
#     plt.title("different h")
#     plt.gca().legend(('0.5', '0.25', '0.2', '0.1', '0.01'))
#     plt.xlabel("x"); plt.ylabel("derivative"); plt.show()

def q1():
    x = 2
    hs = []
    df4 = []
    dff = []
    dfb = []
    h = 1
    while h > 0.0001:
        hs.append(np.log(h))
        df4.append(der_h4(x, h))
        temp = der_h1(x, h)
        dff.append(temp[0])
        dfb.append(temp[1])
        h /= 2

    # comparing different h
    plt.plot(hs, df4); plt.plot(hs, dff); plt.plot(hs, dfb); 
    plt.title("derivative vs log h")
    plt.gca().legend(('O(h^4)', 'forward O(h)', 'backward O(h)'))
    plt.xlabel("log h"); plt.ylabel("derivative"); plt.show()

        
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

def adaptive(a, b, epsilon=2, i=0, f=f2):
    i += 1
    c = (a + b) / 2

    I = (f(a) + f(b) + 4 * f((b + a) / 2))*((b-a)/6)
    if i > 10 : return I
    I_adap = (f(a) + 4 * f((a+c)/2) + 2*f(c) + 4*f((b+c)/2) + f(b))*((b-a)/12)
    
    converged = abs(I - I_adap) < epsilon

    if not converged:
        I_adap = adaptive(a, c, epsilon/2, i) + adaptive(c, b, epsilon/2, i)
    
    return I_adap

def q2():
    for fn in [trapezoid, simpson, gauss_2, gauss_3, adaptive]:
        start = time.time()
        print(fn(0, 2))
        end = time.time()
        print("time taken for {} : {}".format(str(fn) ,end-start))

q2()