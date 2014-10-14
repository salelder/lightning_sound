import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

def stepv(u, v, dt = 1, dx = 1):
    """Steps the advective wave equation with initial data u forward by a timestep dt assuming spacing dx and speed v using FTCS
    
    Keyword arguments:
    u -- initial value array
    v -- speed factor in the advective equation
    dt -- time step (default 1)
    dx -- grid size (default 1)
    """
    dudx = np.gradient(u, dx)
    return u -v*dudx*dt

def stepn(u, v, dt=1, dx=1):
    """Steps the advective wave equation with initial data u forward by a timestep dt assuming spacing dx and speed v using Lax FTCS
    
    Keyword arguments:
    u -- initial value array
    v -- speed factor in the advective equation
    dt -- time step (default 1)
    dx -- grid size (default 1)
    """

    if (abs(v)*dt/dx > 1):
        print("Courant condition not satisfied")
    dudx = np.gradient(u, dx)
    nu = np.zeros(u.size)
    for i in range(u.size-2):
        ui = (u[i+2]+u[i])/2.
        nu[i+1] = ui - v*dt*dudx[i+1]
    nu[0] = u[0]-v*dt*dudx[0]
    nu[u.size-1] = u[u.size-1]-v*dt*dudx[u.size-1]
    return nu

def iteratev(u, v, n, dt=1, dx=1):
    """Plots n successive time steps of the advective equation with initial value u and speed v using FTCS."""
    x = np.linspace(0, (u.size-1)*dt, u.size)
    plt.plot(x, u)
    for i in range(n):
        u = stepv(u, v, dt, dx)
        plt.plot(x, u)

def iteraten(u, v, n, dt=1, dx=1):
    """Plots n successive time steps of the advective equation with initial value u and speed v using Lax FTCS."""    
    x = np.linspace(0, (u.size-1)*dt, u.size)
    plt.plot(x, u)
    for i in range(n):
        u = stepn(u, v, dt, dx)
        plt.plot(x, u)
