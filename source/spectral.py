import numpy as np
import scipy as sp

def basis(n, k, m):
    """Return the value of the nth basis function e^{i2*pi*nk/m} at position k out of m."""
    return np.exp(1j*2*np.pi*n*k/m)

def spectral(f, v, dx, dt):
    """Return a function of position and time (according to the provided time and position step size) built from exponentials represented by the fourier coefficients f approximating the solution to the wave equation with speed v."""
    def func(k, j):
        s = 0
        m = f.size
        for i in range(m):
            s += f[i]/m*basis(i, k - v*j*dt/dx, m)
        return s
    return func
