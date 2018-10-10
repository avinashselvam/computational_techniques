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
        
q1()
