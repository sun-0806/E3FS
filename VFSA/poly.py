import numpy as np
from pypbc import *

def add_poly(L1,L2): 
    R=[]
    if len(L1)>len(L2):
        L1,L2=L2,L1
    i=0
    while i<len(L1):
        R.append(L1[i]+L2[i])
        i+=1
    R=R+L2[len(L1):len(L2)]
    return R

def multiply_poly(L1,L2,pairing):
    if len(L1)>len(L2):
        L1,L2=L2,L1
    zero=[];R=[]
    for i in L1:
        T=zero[:]
        for j in L2:
            T.append(i*j)
        R=add_poly(R,T)
        zero=zero+[Element.zero(pairing, Zr)]
    return R

def divide_poly(L1,L2,pairing):
    if len(L1)<len(L2):return 0,L1
    d=len(L1)-len(L2)
    T=L1[:]
    R=[]
    for i in range(d+1):
        n=T[len(T)-1]/L2[len(L2)-1]
        R=[n]+R
        T1=[Element.zero(pairing, Zr)]*(d-i)+[n]
        T2=multiply_poly(T1,L2,pairing)
        T=subtract_poly(T,T2)
        T=T[:len(T)-1]
    return R,T

def subtract_poly(L1,L2): 
    L2=L2[:]
    for i in range(len(L2)):
        L2[i]=-L2[i]
    return(add_poly(L1,L2))

def polyline(off, scl):
    if scl != 0:
        return np.array([off, scl])
    else:
        return np.array([off])

def poly_self(roots):
    if len(roots) == 0:
        return np.ones(1)
    else:
        roots.sort()
        p = [polyline(-r, 1) for r in roots]
        n = len(p)
        while n > 1:
            m, r = divmod(n, 2)
            tmp = [np.polymul(p[i], p[i+m]) for i in range(m)]
            if r:
                tmp[0] = np.polymul(tmp[0], p[-1])
            p = tmp
            n = m
        return p[0]
