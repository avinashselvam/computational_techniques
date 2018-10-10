from math import sin, cos, pi, log, floor
import matplotlib.pyplot as plt

def q1():
    N = input("\nEnter the number : ")
    a = int(input("Enter the base : "))

    holds_fractional_ans = []
    holds_integer_ans = []

    n = N

    integer, frac = N.split('.')

    integer = int(integer)
    frac = int(frac[::-1])

    print("integer part is ", integer)
    print("fractional part is ", frac)



    frac_in_dec = 0.0
    i = 1

    # only do when there is a fractional part
    if frac != 0:
        # convert to decimal
        while frac != 0:
            frac_in_dec += (frac % 10) / (a ** i)
            i += 1
            frac //= 10

        # upto 5 significant digits
        # decimal to binary of fraction
        while len(holds_fractional_ans) < 5:
            frac_in_dec *= 2
            if frac_in_dec == 1:
                holds_fractional_ans.append(1)
                break
            holds_fractional_ans.append(int(frac_in_dec))
            frac_in_dec -= int(frac_in_dec)

    int_in_dec = 0
    i = 0

    # only do when there is a integer part
    if integer != 0:
        # convert to decimal
        while integer > 0:
            int_in_dec += (integer % 10) * (a ** i)
            i += 1
            integer //= 10

        # decimal to binary of integer
        while int_in_dec > 0:
            holds_integer_ans.append(int_in_dec % 2)
            int_in_dec //= 2

    print("The number {} base {} in base 2 is\n".format(n, a))
    print(''.join(map(str, holds_integer_ans[::-1])), '.', ''.join(map(str, holds_fractional_ans)))

def q2():
    a, b, c = list(map(float, input("\nEnter the coefficients : ").split(' ')))
    
    det = b ** 2 - 4 * a * c

    x1_a = (-1 * b - det ** 0.5) / 2 * a
    x2_a = (-1 * b - det ** 0.5) / 2 * a

    x1_b = -2 * c / ( b - det ** 0.5)
    x2_b = -2 * c / ( b + det ** 0.5)

    # print(x1_a, x1_b, x2_a, x2_b)

    values = []

    for each in [x1_a, x1_b, x2_a, x2_b]:
        values.append(each)
        print("\nError at root {} is {}".format(each, abs(a * (each ** 2) + b * each + c)))
    
    if values[0] > values[2]: x1 = x2_a
    else: x1 = x1_a

    if values[1] > values[3]: x2 = x2_b
    else: x2 = x1_b

    print("\nThe chosen roots are {} and {}\n".format(x1, x2))

def q3():
    x = 0.1
    value = 1./(1-x)

    ans = 1
    i = 0

    error = abs(ans - value)
    errors = [error]

    evaluations = [ans]

    while error > 10 ** -5:
        i += 1
        ans += x ** i
        evaluations.append(ans)
        error = abs(ans - value)
        errors.append(error)


    print("The values of 1/1-x for x=0.1 at different orders of expansion are {}".format(evaluations))
    print("The final error is {}".format(error))

    relative_errors = [abs(i - j) for i, j in zip(evaluations[:-1], evaluations[1:])]
    print(relative_errors)

    plt.plot(range(len(errors)), errors, color='blue')
    plt.plot(range(len(relative_errors)), relative_errors, color='red')
    plt.show()

def q4():

    at = pi / 3
    h = pi / 3

    res = []
    error = []
    hs = []

    def calc(h):
        x = at - h
        return cos(x) - h*sin(x) + (h**2)*cos(x)/2

    while h != pi / 96:
        val = calc(h)
        res.append(val)
        error.append(log(abs(val - 0.5)))
        hs.append(log(h))
        h /= 2

    print(res, hs)

    plt.plot(hs, error)
    plt.xlabel("Value of h")
    plt.ylabel("absolute error")
    plt.show()

q1()