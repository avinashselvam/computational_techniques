from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm, cond, solve
from scipy.linalg import lu

MAX_ITER = 1000

def relax(x_new, h, x_old):
    return x_new * h + x_old * (1 - h)

def sig_digits(a):
    return float(str(a)[:7])
    
# used to float every entry
f = np.vectorize(sig_digits)

def plot_h():
    STEP_SIZE = 0.2

    hs = []
    gs = []
    js = []

    for h in np.arange(0.2, 2, STEP_SIZE):
        a, b, c = h, gauss_seidel(cs2, b2, 0.01, h)[1], jacobi(cs1, b1, 0.01, h)[1]
        hs.append(a)
        gs.append(b)
        js.append(c)
    plt.plot(hs, gs)
    plt.plot(hs, js)
    plt.gca().legend(('Gauss','Jacobi'))
    plt.xlabel("h value")
    plt.ylabel("iterations")
    plt.show()

def jacobi(cs, b, epsilon = 0.000001, h=0.5, use_relaxation=True):

    cs = f(cs)
    b = f(b)

    # initial assumption
    x = np.array([0, 0, 0])
    x = f(x)

    x_ = np.zeros(3)

    # counter
    i = 0

    rel_error = 100

    while rel_error > epsilon:

        i += 1
        if i > MAX_ITER: break

        x_prev = np.copy(x)

        if not use_relaxation:
            x1 = -(cs[0,1] * x[1] + cs[0,2] * x[2] - b[0]) / cs[0,0]
            x2 = -(cs[1,0] * x[0] + cs[1,2] * x[2] - b[1]) / cs[1,1]
            x3 = -(cs[2, 0] * x[0] + cs[2, 1] * x[1] - b[2]) / cs[2, 2]

            # for any matrix
            # for k in range(3):
            #     cs_without_k = list(cs[k,:]).pop(k)
            #     x_without_k = list(x).pop(k)
            #     xk = (b[k] - np.dot(cs_without_k, x_without_k)) / cs[k, k]
            #     x_.append(xk)

        else:
            x1 = -(cs[0, 1] * x[1] + cs[0, 2] * x[2] - b[0]) / cs[0, 0]
            x1 = relax(x1, h, x[0]) if i else x1
            x2 = -(cs[1, 0] * x[0] + cs[1, 2] * x[2] - b[1]) / cs[1, 1]
            x2 = relax(x2, h, x[1])  if i else x2
            x3 = -(cs[2, 0] * x[0] + cs[2, 1] * x[1] - b[2]) / cs[2, 2]
            x3 = relax(x3, h, x[2]) if i else x3
            
            # for any size
            for k in range(3):
                cs_without_k = list(cs[k,:]).pop(k)
                x_without_k = list(x_).pop(k)
                xk = (b[k] - np.dot(cs_without_k, x_without_k)) / cs[k, k]
                xk = relax(xk, h, x_[k]) if i else xk
                x_[k] = xk
        
        x = np.array([x1, x2, x3])

        rel_error = abs(norm(x - x_prev))
        # print(x, x_)

    print(x)
    print("number of iterations : {}".format(i))
    print("terminated with relative error : {}\n".format(rel_error))

    return x, i



def gauss_seidel(cs, b, epsilon = 0.000001, h=0.5, use_relaxation=True):

    cs = f(cs)
    b = f(b)

    x = np.array([0, 0, 0, 0])
    x = f(x)

    i = 0

    rel_error = 100

    while rel_error > epsilon:

        i += 1
        if i > MAX_ITER: break

        x_prev = np.copy(x)
        
        if not use_relaxation:
            x[0] = -(cs[0, 1] * x[1] + cs[0, 2] * x[2] + cs[0,3]*x[3] - b[0]) / cs[0, 0]
            x[1] = -(cs[1, 0] * x[0] + cs[1, 2] * x[2] + cs[1,3]*x[3] - b[1]) / cs[1, 1]
            x[2] = -(cs[2, 0] * x[0] + cs[2, 1] * x[1] + cs[2,3]*x[3] - b[2]) / cs[2, 2]
            x[3] = -(cs[3, 0] * x[0] + cs[3, 1] * x[1] + cs[3,2]*x[3] - b[3]) / cs[3, 3]

        else:
            x1 = -(cs[0, 1] * x[1] + cs[0, 2] * x[2] + cs[0, 3] * x[3] - b[0]) / cs[0, 0]
            x[0] = relax(x1,h,x[0])  if i else x1
            x2 = -(cs[1, 0] * x[0] + cs[1, 2] * x[2] + cs[1, 3] * x[3] - b[1]) / cs[1, 1]
            x[1] = relax(x2,h,x[1])  if i else x2
            x3 = -(cs[2, 0] * x[0] + cs[2, 1] * x[1] + cs[2, 3] * x[3] - b[2]) / cs[2, 2]
            x[2] = relax(x3,h,x[2])  if i else x3
            x4 = -(cs[3, 0] * x[0] + cs[3, 1] * x[1] + cs[3, 2] * x[3] - b[3]) / cs[3, 3]
            x[3] = relax(x4, h, x[3]) if i else x4
    
            x = np.array(x)

            rel_error = abs(norm(x - x_prev))
        
    print(x)
    print("number of iterations : {}".format(i))
    print("terminated with relative error : {}\n".format(rel_error))

    return x, i

def backward_sub(A, b):

    n = np.shape(A)[0]
    x = [b[n-1]/A[n-1,n-1]]

    for j in range(1, n):
        i = n - j - 1
        xi = (b[i] - np.dot(A[i,::-1][:j], np.array(x)))/ A[i, i]
        x.append(xi)

    return x[::-1]

def forward_sub(A, b):

    n = np.shape(A)[0]
    x = [b[0]/A[0,0]]

    for i in range(1,n):
        xi = (b[i] - np.dot(A[i,:i], np.array(x))) / A[i, i]
        x.append(xi)

    return x

def LU_factor(pivot = True):

    cond_threshold = 5

    A = np.array([[2., -1., 3., 2.], [2., 2., 0., 4.], [1., 1., -2., 2.], [1., 3., 4., -1.]])
    A = f(A)

    org = np.copy(A)

    b = [1., 2., 3., 4.]

    n = A.shape[0]
    I = np.eye(n)
    
    for i in range(n):

        ### pivoting code not used
        if pivot:
            max_index = np.argmax(np.abs(A[i:, i])) + i
            # swap rows at i and max_index
            if i != max_index:
                temp = np.copy(A[i,:])
                A[i,:] = np.copy(A[max_index,:])
                A[max_index,:] = np.copy(temp)

                temp = b[i]
                b[i] = b[max_index]
                b[max_index] = temp
        
        # transform I matrix to L matrix
        I[i+1:, i] = A[i+1:, i] / A[i,i]
        
        # transform A matrix to U matrix
        for j in range(i+1, n):
            factor = A[j,i]/A[i,i]
            A[j, i:] -= factor * A[i, i:]
    
    cond_A = cond(A, p=np.inf)

    if cond_A > cond_threshold:
        print("condition number is greater than threshold. Aborting...")
        ans = -1
    else:
        print("cond no : ", cond_A)
        L, U = I, A

        b1 = forward_sub(L, b)
        ans = backward_sub(U, b1)

    print("\nL matrix : \n")
    print(L)
    print("\nU matrix : \n")
    print(U)
    print("\nAns to Ax = b : \n")
    print(ans)
    print("\nFor verification calc Ax - b : \n")
    print(np.matmul(org, ans) - b)

# values for 1st question
cs1 = np.array([[4, 1, -1], [1, -5, -1], [2, -1, -6]])
b1 = np.array([13, -8, -2])

# values for 2nd question
cs2 = np.array([[1, 1 / 2, 1 / 3, 1 / 4], [1 / 2, 1 / 3, 1 / 4, 1 / 5], [1 / 3, 1 / 4, 1 / 5, 1 / 6], [1 / 4, 1 / 5, 1 / 6, 1 / 7]], dtype=np.float16)
b2 = np.array([25 / 12, 77 / 60, 57 / 60, 319 / 420])


# jacobi(cs1, b1)
# gauss_seidel(cs2, b2)
LU_factor(pivot=True)
# plot_h()