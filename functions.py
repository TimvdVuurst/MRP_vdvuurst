import numpy as np

def Romberg(a,b,func,m=6):
    """Romberg's algorithm for integration using Richardson extrapolation.

    Args:
        a (float): Lower bound of the integral (incl).
        b (float): upper bound of the integral (incl).
        func (method): function to integrate. Must have only one callable variable which is the one we integrate over.
        m (int, optional): Order of extrapolation. Defaults to 6.

    Returns:
        tuple of floats: (estimate of the integral, estimated error on integral by absolute difference between current and previous best guess)
    """

    h = (b-a) 
    r = np.zeros(m+1) #this array will hold our estimates
    r[0] = 0.5 * h * (func(a) + func(b)) #first guess
    Np = 1
    for i in range(1,m+1): #loop over rows
        h *= 0.5
        x = a + h #point to evaluate the function at, a little ahead of the previous
        for _ in range(Np):
            r[i] += func(x)
            x += 2*h
    
        r[i] = 0.5 * r[i-1] + h*r[i]

        Np *= 2

    Np4 = 1
    for i in range(m+1): #loop over columns
        Np4 *= 4
        for j in range(0,m - i): #loop over relevant rows
            r[j] = (Np4 * r[j+1] - r[j]) / (Np4 - 1) #update rule for Richardson extrapolation. Np4 = 2^(2j) = 4^j


    return r[0],np.abs(r[0] - r[1])


def modified_logspace(start, stop, num, base = 10):
    res = np.zeros(num)
    stop = np.log10(stop)
    if start == 0:
        start = np.log10(0.1)
        y = np.power(base, np.linspace(start, stop, num - 1))
        res[1:] = y
    else:
        start = np.log10(start)
        res = np.power(base, np.linspace(start, stop, num))

    return res