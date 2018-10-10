import matplotlib.pyplot as plt
import numpy as np

# xs = list(map(float, input("Enter all x").split(" ")))
# ys = list(map(float, input("Enter all y").split(" ")))

xs = [4, 5, 6, 7, 8]
ys = [2.0, 2.3607, 2.44949, 2.64575, 2.828483]

n = len(xs)

def lagrange_p(x):
    num = 1
    den = 1

    ps = []
    fx = 0

    for x1 in xs:
        for each in xs:
            if each == x1:
                continue
            num *= (x - each)
            den *= (x1 - each)
        ps.append(num / den)
    for yi, pi in zip(ys, ps):
        fx += yi*pi
    ps.append(fx)
    
    return ps

def newton_p(x):
    
    ps = [1]

    fx = 0

    for x1 in xs:
        p = ps[-1]*(x - x1)
        ps.append(p)
    for yi, pi in zip(ys, ps):
        fx += yi*pi
    ps.append(fx)
    return ps
        
            

# print(p(0))
x_tests = np.arange(4, 8, 0.01)
plt.title("Lagrange polynomial")
plt.plot(x_tests, [lagrange_p(i) for i in x_tests])
plt.show()

plt.title("Newton polynomial")
plt.plot(x_tests, [newton_p(i) for i in x_tests])
plt.show()