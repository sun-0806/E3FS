from pypbc import *
import numpy as np
from poly import poly_self,subtract_poly,multiply_poly,add_poly,divide_poly

def Acc(P,pk,pairing):
    acc=Element.one(pairing, G2) #Initializes an element in G2 to 1
    for i in range(len(P)):
        v=Element(pairing, G2, value =pk[i]**P[i])
        acc=Element(pairing, G2, value = acc*v)
    return acc

def ext_euclid(f, h,pairing):
    '''
    The extended Euclidean algorithm finds x,y such that xf+yh=gcd(f,h)
    The highest degree of f is greater than or equal to h
    '''
    if (len(h) == 1 and h[0]==Element.zero( pairing, Zr)) or len(h) == 0:
        return [Element.one(pairing, Zr)], [Element.zero( pairing, Zr)], f
    else:
        p, r = divide_poly(f,h,pairing)
        x, y, coef = ext_euclid(h, r,pairing)
        x,y=y, subtract_poly(x, multiply_poly(p, y,pairing))
        return x, y, coef

def Proved_disjoint(P1_Zr,P2_Zr,pk,pairing):
    '''Find the polynomial Q1,Q2 satisfies P1_Zr*Q1+P2_Zr*Q2=gcd(P1_Zr,P2_Zr)'''
    Q1,Q2,coef=ext_euclid(P1_Zr,P2_Zr,pairing)
    #print('Q1,Q2,coef=',Q1,Q2,coef)
    if len(coef) != 1:
        exit()
    else:
        pi_1=Acc(Q1,pk,pairing)
        pi_2=Acc(Q2,pk,pairing)
        return pi_1,pi_2,coef[0]


def Verify_disjoint(acc1, acc2, pi_1, pi_2, coef, pairing,g):
     '''Multi-set accumulator verification algorithm'''
    right = Element(pairing, GT, value=pairing.apply(g,g) ** coef)
    left = Element(pairing, GT, value=pairing.apply(acc1, pi_1) * pairing.apply(acc2, pi_2))
    if left == right:
        return 1
    else:
        return 0

def keyGen(m):
    q_1 = get_random_prime(60)
    q_2 = get_random_prime(60)
    params = Parameters(n=q_1*q_2)
    pairing = Pairing(params)  
    g = Element.random(pairing, G2)  
    s = Element.random(pairing, Zr)
    q = m+1  
    pk = []  
    for i in range(q):
        pk.append(Element(pairing, G2, value=g ** (s**i)))
    return g, pk, pairing

def process(X, pairing):
    P = poly_self(list(np.array(X))) # Given the root X of the polynomial, find the coefficient P
    P_Zr = [Element(pairing, Zr, value=int(item)) for item in P] # Convert coefficients to elements in Zr
    return P_Zr
