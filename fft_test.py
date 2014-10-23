import numpy as np
from scipy.fftpack import fft, ifft
import matplotlib.pyplot as plt

t = arange(0,2*pi,.1) # 63 elements
x = sin(t) + sin(3*t) #+ sin(10*t)
y = fft(x)
#plot(x)
#plot(y)

df = 1/6.3
def mode (k):
    # k is the index in y. Returns the wave for the k-th FFT mode.
    wavek = array(t)
    time = 0
    c = y[k]
    for i in wavek:
        wavek[time] = c*(e**(2*pi*1j*k*df*i))
        time += 1
    return wavek

def rebuild():
    # rebuild the function from the FFT
    wave = y[0]*ones(63)
    for k in arange(1,31,1):
        wave += mode(k)
    return wave
    
def solnt(t):
    # returns the solution to the wave equation at time t
    ct = array(y)
    for n in range(63):
        ct[n] = y[n]*cos(n*t/63)
    return ifft(ct)

#plot(rebuild()/63)
for t in range(0,1,100):
    plot(solnt(t))
    savefig("filename"+str(t)+".png")
    clf()