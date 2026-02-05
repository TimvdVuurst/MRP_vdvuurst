from numpy import full_like, exp, power

def constant_func(x,a):
    return full_like(x, a)

def linear_func(x, m, c):
    return m * x + c

def parabola_func(x, a, b, c):
    return a * x**2 + b * x + c

def exponential_func(x, A, B, C):
    return A * exp(-B * x) + C

def exponential_squared_func(x, A, B, C):
    return A * exp(-B * x**2) + C

def power_law_func(x, p, n, q):
    return p * power(x, n) + q

def power_linear_func(x, p, n, q, b):
    return p * power(x, n) + q * x + b

def inverse_func(x, a, b):
    return a/x + b