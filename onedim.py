import numpy as np
import matplotlib.pyplot as plt

L= 1.
v= 1.
dt= .01

# f is initial displacement, g is initial transverse velocity
x= arange(0,L,.01)
f= sin(2.*pi*x/L)
g= 0.*x
  
def c(n,h):
  # computes c_n for F-series with complex coefficients.
  # h is an array of samples
  cn= 0.+0.j
  N= len(h)
  dx= L/N
  for k in range(N):
    cn+= h[k] * np.exp(-1j*(2*pi*n/L)*k*dx) * dx
  return cn

def AB(c,d,n):
  # returns a list [A_n,B_n] given c_n and d_n
  z= 1j*2*pi*n*v/L
  B= (c-d/z)/2
  A= c-B
  if n!=0:
    return [A,B]
  else:
    return [c/2,c/2]

def u(x,t,N):
  # sum from -N to N approximates integral from -infinity to infinity
  uxt= 0.+0.j
  for n in range(-N,N):
    cn= c(n,f)
    dn= c(n,g)
    An, Bn= AB(cn,dn,n)
    uxt+= np.exp(1j*2*pi*n*x/L) * (An*np.exp(1j*2*pi*n*v*t/L) + Bn*np.exp(-1j*2*pi*n*v*t/L))
  return uxt  

def U(t,N):
  # returns the array of the spatial function for fixed t (real part only)
  Ut= array(x)
  for k in range(len(x)):
    Ut[k]= u(x[k],t,N)
  return Ut

for t in arange(0.,1.,.1):
  plot(x,U(t,10))