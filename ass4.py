import numpy as np
import matplotlib.pyplot as plt

def f1(x):
    return np.sin(10 * x) + np.cos(3 * x)
    
def f2(x):
    # return (x - 1) * (np.exp(x - 1) - 1)
    return (x-1)**2

def f2_(x):
    # return x * np.exp(x - 1) - 1
    return 2*(x-1)

def rate_of_convergence(error_in_p):
    y = []
    for i in range(len(error_in_p) - 1):
        np.log(error_in_p[i+1])/np.log(error_in_p[i])

def plot_fn(f, a, b):
    x = np.arange(a, b, 0.01)
    y = [f(i) for i in x]
    plt.plot(x, y)
    plt.plot(x, [0 for i in x])
    plt.title("plot of the function")
    plt.show()
    
def plot_err(errors):
    plt.plot(range(len(errors)), np.log(errors))
    plt.title("errors vs iteration")
    plt.xlabel("iterations")
    plt.ylabel("log error")
    plt.show()

def search_ab(xs):
    brackets = []
    fxs = [f1(i) for i in xs]
    for i in range(len(xs)-1):
        if fxs[i] * fxs[i+1] < 0:
            brackets.append((xs[i], xs[i + 1]))
    
    return brackets


a, b = 3, 6
step = 0.01

xs = np.arange(a, b + step, step)

epsilon = 0.00001
delta = 0.00001
c_prev = 0

def bisection(a, b):

    errors = []
    c_prev = 0
    converged = False
    
    # termination conditions i.e c, f(c) is within the rectangle
    while not converged:

        # get mid point
        c = (a + b) / 2

        # needed for plotting
        error = abs(f1(c))
        diff = abs(c - c_prev)
        errors.append(error)

        # for next point
        if f1(c) * f1(a) < 0: b = c
        elif f1(c) * f1(b) < 0: a = c
        else: break

        c_prev = c
        converged = (diff < delta) or (error < epsilon)
    
    # plot_err(errors)
    return c, errors

def regula_falsi(a, b):

    errors = []
    converged = False
    c_prev = 0
    i = 0

    # termination conditions i.e c, f(c) is within the rectangle
    while not converged or i < 100:
        i+=1
        # get x intercept of line joining a, f(a) and b, f(b)
        c = b - (f1(b) * (b - a)) / (f1(b) - f1(a))
        
        # needed for plotting
        error = abs(f1(c))
        diff = abs(c - 3.25867761223)
        errors.append(diff)

        # for next point
        if f1(c) * f1(a) < 0: b = c
        elif f1(c) * f1(b) < 0: a = c
        else: break
        
        c_prev = c
        converged = (diff < delta) or (error < epsilon)

    # plot_err(errors)
    return c, errors


    
def newton_raphson(pk_1):

    errors = []
    iteration = 0
    p = 1

    converged = False
    
    while not converged or iteration < 10000:

        iteration += 1
        if f2_(pk_1) == 0: break
        
        pk = pk_1 - (f2(pk_1) / f2_(pk_1))
        pk_1 = pk

        # errors.append(abs(f2(pk)))
        errors.append(abs(p - pk))
        
        converged = errors[-1] < epsilon
        
    plot_err(errors)
    
    return pk, errors

def secant(pk_1, pk_2):

    errors = [100]
    iteration = 0
    p = 1

    converged = False

    while not converged or iteration < 10000:
        
        iteration += 1
        if f2(pk_1) - f2(pk_2) == 0: break
        
        pk = pk_1 - f2(pk_1) * (pk_1 - pk_2) / (f2(pk_1) - f2(pk_2))
        pk_2, pk_1 = pk_1, pk
        
        # errors.append(abs(f2(pk)))
        errors.append(abs(p - pk))
        
        converged = errors[-1] < epsilon
    
    plot_err(errors)

    # print("iterations : {}".format(iteration))
    # print("solution : {}".format(pk))

    return pk, errors

def q1():
    plot_fn(f1, 3 , 6)
    brackets = search_ab(xs)
    for a, b in brackets:
        bis, reg = bisection(a, b), regula_falsi(a, b)
        l1, l2 = range(len(bis[1])), range(len(reg[1]))
        print("\nthe root between {} and {} by".format(a, b))
        print("Bisection : {}".format(bis[0]))
        print("Regula falsi : {}\n".format(reg[0]))
        plt.plot(l1, bis[1])
        plt.plot(l2, reg[1])
        plt.gca().legend(('bisection', 'regula'))
        plt.show()

def q2():
    plot_fn(f2, -1, 2)
    p0, p1 = -2, -1
    print("\nNewton raphson : {}".format(newton_raphson(p0)))
    print("Secant : {}\n".format(secant(p0, p1)))

# q1()
# q2()

# print(regula_falsi(3, 3.25)[0])
# err = regula_falsi(-2, -1)[1]

err = secant(0, 2)[1]
# err = newton_raphson(-2)[1]

for r in np.arange(1,2.25,0.25):
    ratio = [np.log(err[i + 1]) - r*np.log(err[i]) for i in range(len(err) - 1)]
    plt.plot(range(len(ratio)), ratio)
plt.gca().legend(('1', '1.25', '1.5', '1.75', '2.0'))
plt.xlabel("iterations")
plt.ylabel("log K")
plt.show()