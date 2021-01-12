#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cupy as cp
import numpy as np
from cupy import random
import scipy as sc
from scipy import linalg
import matplotlib as plt
import numpy as np
from numpy import genfromtxt
from scipy.stats import beta
from scipy.stats import norm
from scipy.stats import gamma
from scipy.stats import invgamma
import matplotlib.pyplot as plt
from sympy import Matrix
import seaborn as sns
import math
import pandas as pd
ordqz = sc.linalg.ordqz
svd = sc.linalg.svd
m = cp.matmul
resh = cp.reshape


# In[2]:


sigma = 1.2
k = .82
phi_x = .5
phi_pi = 1.5
p_q = .5
p_r = .5
p_u = .5
A0 = cp.array(((1,0,sigma),(-k,1,0),(-phi_x,-phi_pi,1)))
A1 = cp.array(((sigma,0,0),(0,1,0),(0,0,1)))
A2 = cp.array(((sigma,0,0),(0,0,0),(0,0,1)))
R = cp.array(((p_r,0,0),(0,p_u,0),(0,0,p_q)))
T = 500
Z = cp.zeros((3,T))


# In[3]:


PHI = cp.random.standard_normal((12,1))
PHI0 = cp.zeros((3,1))
PHI1 = cp.zeros((3,3))
r = cp.random.rand()
u = cp.random.rand()
q = cp.random.rand()


# In[4]:


I = cp.zeros((12,3))
I[0,0] = 1
I[1,1] = 1
I[2,2] = 1

U = cp.zeros((12,3))
Rs = cp.zeros((12,3))
Q = cp.zeros((12,3))

U[3,0] = 1
Rs[4,0] = 1
Q[5,0] = 1
U[6,1] = 1
Rs[7,1] = 1
Q[8,1] = 1
U[9,2] = 1
Rs[10,2] = 1
Q[11,2] = 1
sig_r = 1.6**.5 
sig_u = 1.6**.5
sig_q = 1.6**.5


# In[5]:


T = 500
#EZ = PHI_0+m(PHI_1,shocks) 
#EZ1 = PHI_0 +  m(m(PHI_1,R),shocks)
RR = 10*cp.random.standard_normal([12,12])
Z[:,0] = 100*cp.random.standard_normal([3])
eps_r = cp.random.standard_normal([T])
eps_u = cp.random.standard_normal([T])
eps_q = cp.random.standard_normal([T])
eps_r[0] = abs(cp.random.standard_normal())
PHI = cp.random.standard_normal([12,1])
r = 0
u = 0
q = 0
PHI_old = PHI
for t in range(T):
    r = p_r*r+sig_r*eps_r[t]
    u = p_u*r+sig_u*eps_u[t]
    q = p_q*r+sig_q*eps_q[t]
    shocks = cp.vstack((r,u,q))
    PHI0 = resh(PHI[0:3],[3,1])
    PHI1 = resh(PHI[3:],[3,3])
    PHI_old = PHI   
    EZ1 = PHI0 +  m(m(PHI1,R),shocks)
    Z[:,t] = cp.reshape(m(cp.linalg.inv(A0),(cp.reshape(m(A1,shocks),[3,1])+m(A2,EZ1))),[3])
    if Z[2,t] <0:
        Z[2,t] = 0    
    X = I+r*Rs+u*U+q*Q 
    EZ =  PHI0 +  m(PHI1,shocks)
    err = resh(Z[:,t],[3,1]) - EZ
    RR = RR+.02*(m(X,X.T)-RR)
    PHI = PHI+.02*m(m(cp.linalg.inv(RR),X),err)
    if cp.any(abs(cp.linalg.eigvalsh(PHI1))>1) == True:
                                            PHI = PHI_old
        


# In[6]:


plt.plot(cp.asnumpy(Z[2,:]))


# In[7]:


RR = 10*cp.random.standard_normal([12,12])
Z[:,t] = 100*cp.random.standard_normal([3])
eps_r = cp.random.standard_normal([T])
eps_u = cp.random.standard_normal([T])
eps_q = cp.random.standard_normal([T])
eps_r[0] = abs(cp.random.standard_normal())
r = 0
u = 0
q = 0
n = 0
sig = 0
err = cp.zeros([3,1])
PHI = cp.random.standard_normal([12,1])
for t in range(T):
    n = t+3
    r = p_r*r+sig_r*eps_r[t]
    u = p_u*r+sig_u*eps_u[t]
    q = p_q*r+sig_q*eps_q[t]
    shocks = cp.vstack((r,u,q))
    PHI0 = resh(PHI[0:3],[3,1])
    PHI1 = resh(PHI[3:],[3,3])
    EZ1 = PHI0 +  m(m(PHI1,R),shocks)
    sig = (1/(n-2))*(sig*(n-3)+(err[2]**2))
    if t == 0:
        sig = 1
    if EZ1[2]<0:
        lamb_0 = sc.stats.truncnorm.mean(abs(cp.float(EZ[2])),math.inf,0,1)
        #lamb_0 = expec_TN(lower_bound =cp.float(EZ[2]) ,variance=sig)
        EZ1[2] = EZ1[2]+lamb_0
    Z[:,t] = cp.reshape(m(cp.linalg.inv(A0),(cp.reshape(m(A1,shocks),[3,1])+m(A2,EZ1))),[3])
    
    if Z[2,t] <0:
        Z[2,t] = 0     
    X = I+r*Rs+u*U+q*Q 
    EZ =  PHI0 +  m(PHI1,shocks)
    err =  resh(Z[:,t],[3,1]) - resh(EZ,[3,1])
    RR = RR+.02*(m(X,X.T)-RR)
    PHI = PHI+.02*m(m(cp.linalg.inv(RR),X),err)
    if cp.any(abs(cp.linalg.eigvalsh(PHI1))>1) == True:
                                            PHI = PHI_old
  
    


# In[8]:


sig


# In[9]:


plt.plot(cp.asnumpy(Z[1,:]))


# In[10]:


USA_GDP = pd.read_csv(r"C:\Research\First Glance at Data\USoutput.csv")
y1 = USA_GDP[USA_GDP['DATE']>= '1961-01-01']['2019-01-01'>=USA_GDP['DATE']];
y_data1 =np.array(y1)[:,1]

x_1 = cp.zeros([232])
i = 0
while i<=231:
    x_1[i] = (float(y_data1[i])-float(y_data1[i-1]))/float(y_data1[i-1])

    i= i+1
plt.plot(cp.asnumpy(x_1[1:]))


# In[ ]:





# In[4]:


################  Habit Formation   #############################


# In[5]:


def gensys(G0, G1, PSI, PI, DIV=1 + 1e-8,
           REALSMALL=1e-6,
           return_everything=False):
    """
    Solves a Linear Rational Expectations model via GENSYS.

    Γ₀xₜ = Γ₁xₜ₋₁ + Ψεₜ + Πηₜ

    Returns
    -------

    RC : 2d int array
         [ 1,  1] = existence and uniqueness
         [ 1,  0] = existence, not uniqueness
         [-2, -2] = coincicdent zeros

    Notes
    -----
    The solution method is detailed in ...

    """
    n, pin = G0.shape[0], PI.shape[1]

    with np.errstate(invalid='ignore', divide='ignore'):
        AA, BB, alpha, beta, Q, Z = ordqz(G0, G1, sort='ouc', output='complex')
        zxz = ((np.abs(beta) < REALSMALL) * (np.abs(alpha) < REALSMALL)).any()

        x = alpha / beta
        nunstab = (x * x.conjugate() < 1.0).sum()

    if zxz:
        RC = [-2, -2]
        #print("Coincident zeros")
        return

    nstab = n - nunstab

    Q = Q.T.conjugate()
    Qstab, Qunstab = Q[:nstab, :], Q[nstab:, :]

    etawt = Qunstab.dot(PI)
    ueta, deta, veta = np.linalg.svd(etawt, full_matrices=False)

    bigev = deta > REALSMALL
    deta = deta[bigev]
    ueta = ueta[:, bigev]
    veta = veta[bigev, :].conjugate().T

    RC = np.array([0, 0])
    RC[0] = len(bigev) >= nunstab

    if RC[0] == 0:
            RC[1] = 0
        #warnings.warn(
         #   f"{nunstab} unstable roots, but only {len(bigev)} "
         #   " RE errors! No solution.")

    if nunstab == n:
        raise NotImplementedError("case nunstab == n, not implemented")
    else:
        etawt1 = Qstab.dot(PI)
        ueta1, deta1, veta1 = svd(etawt1, full_matrices=False)
        bigev = deta1 > REALSMALL
        deta1 = deta1[bigev]
        ueta1 = ueta1[:, bigev]
        veta1 = veta1[bigev, :].conjugate().T

    if veta1.size == 0:
        unique = 1
    else:
        loose = veta1 - veta.dot(veta.conjugate().T).dot(veta1)
        ul, dl, vl = np.linalg.svd(loose)
        unique = (dl < REALSMALL).all()

        # existence for general epsilon[t]
        AA22 = AA[-nunstab:, :][:, -nunstab:]
        BB22 = BB[-nunstab:, :][:, -nunstab:]
        M = np.linalg.inv(BB22).dot(AA22)

    if unique:
        RC[1] = 1
    else:
        pass
        # print("Indeterminancy")

    deta = np.diag(1.0 / deta)
    deta1 = np.diag(deta1)

    etawt_inverseT = ueta.dot((veta.dot(deta)).conjugate().T)
    etatw1_T = veta1.dot(deta1).dot(ueta1.conjugate().T)
    tmat = np.c_[np.eye(nstab), -(etawt_inverseT.dot(etatw1_T)).conjugate().T]

    G0 = np.r_[tmat.dot(AA), np.c_[np.zeros(
        (nunstab, nstab)), np.eye(nunstab)]]
    G1 = np.r_[tmat.dot(BB), np.zeros((nunstab, n))]

    G0i = np.linalg.inv(G0)
    G1 = G0i.dot(G1)

    impact = G0i.dot(np.r_[tmat.dot(Q).dot(
        PSI), np.zeros((nunstab, PSI.shape[1]))])

    G1 = np.real(Z.dot(G1).dot(Z.conjugate().T))
    impact = np.real(Z.dot(impact))

    if return_everything:
        GZ = -np.linalg.inv(BB22).dot(Qunstab).dot(PSI)
        GY = Z.dot(G0i[:, -nunstab:])

        return G1, impact, M, GZ, GY, RC

    else:
        return G1, impact, RC


# In[6]:


def REE_gen(para):
    para = cp.asnumpy(para)
    p_r= float(para[0])
    sigma=float(para[1])
    phi_pi = float(para[2])
    phi_x=float(para[3])
    Beta=.99
    nu=float(para[5])
    theta=float(para[6])
    g=float(para[7])
    sig_r = float(para[8])
    alpha =float(para[9])
    psi = float(para[10])
    h = float(para[11])
    gamma = float(para[12])
    rho = float(para[13])
    p_u = float(para[14])
    sig_e = float(para[15])
    sig_u = float(para[16])
    p_q =float(para[17])
    alpha_p = (alpha/(1-alpha));
    k = ((1-Beta*theta)*(1-theta))/((1+alpha_p*psi)*theta*(1+gamma*Beta*theta));
    c1 = (sigma/((1-h)*(1-h*Beta)));
    c2 = (nu/(alpha + nu));
    w1 = (1+(h**2)*Beta + h*Beta)*((1+h+(h**2)*Beta)**-1)
    w2 =((1-h)*(1-h*Beta))*((sigma*(1+h+h**2*(Beta)))**-1)
    w3 = (-h*Beta/(1+h+(h**2)*Beta));
    w4 =h*((1+h+(h**2)*Beta)**-1)
    n1 = k*c2*c1+k*(h**2)*Beta*c1*c2-k*alpha_p
    n2 = -k*(c2)*(c1)*(h);
    n3 = -k*h*Beta*c1*c2;
    n4 =  Beta*((1+gamma*Beta*theta)**-1);
    n5 = gamma*((1+gamma*Beta*theta)**-1) + (-gamma*psi*alpha_p*k);  
    x_t   = 0;
    pi_t   = 1;
    i_t   = 2;
    r_t  = 3;
    u_t   = 4;
    Ex_t = 5;
    Epi_t = 6;
    Ei_t = 7;
    Ex_t2 = 8;
    Epi_t2 = 9;
    Ei_t2 = 10;
    
    ex_sh  = 0;
    epi_sh  =1;
    ei_sh  = 2;
    ex2_sh  =3 ;
    epi2_sh  =4 ;
    ei2_sh  =5 ;
    
    r_sh = 0;
    pi_sh = 1;
    i_sh = 2;
    
    neq  = 11;
    neta = 6;
    neps = 3;
    GAM0 = np.zeros([neq,neq]);
    GAM1 = np.zeros([neq,neq]);
    C = np.zeros([neq,1]);        
    PSI = np.zeros([neq,neps]);
    PPI = np.zeros([neq,neta]);
    eq_1 = 0
    eq_2    = 1;  
    eq_3    = 2;  
    eq_4    = 3;  
    eq_5   = 4;  
    eq_6    = 5;  
    eq_7    = 6; 
    eq_8    = 7;
    eq_9    = 8;
    eq_10    = 9;
    eq_11    = 10;
#x_t
    GAM0[eq_1,x_t]   =  1;
    GAM0[eq_1,Ex_t]   =  -w1;
    GAM0[eq_1,Epi_t]   =  -w2;
    GAM0[eq_1,i_t]   =  w2;
    GAM0[eq_1,r_t]   =  w2;
    GAM0[eq_1,Ex_t2]   =  -w3;
    GAM1[eq_1,x_t] = w4;
#pi_t
    GAM0[eq_2,pi_t]   =  1;
    GAM0[eq_2,x_t]   = -n1;
    GAM1[eq_2,x_t]   = n2;
    GAM0[eq_2,Ex_t]   = -n3;
    GAM0[eq_2,Epi_t]   = -n4;
    GAM1[eq_2,pi_t]   =  n5;
    GAM1[eq_2,u_t]  =  1;
#i_t
    GAM0[eq_3,x_t]   = -(1-rho)*phi_x;
    GAM0[eq_3,pi_t]  = -(1-rho)*phi_pi;
    GAM0[eq_3,i_t]  =1;
    GAM1[eq_3,i_t]  = rho;
    PSI[eq_3,i_sh] = 1;

#r_t
    GAM0[eq_4,r_t]   = 1;
    GAM1[eq_4,r_t] = p_r;
    PSI[eq_4,r_sh] = 1;
#u_t
    GAM0[eq_5,u_t]   = 1;
    GAM1[eq_5,u_t] = p_u;
    PSI[eq_5,pi_sh] = 1;
#Epi_t

    GAM0[eq_6,pi_t]   = 1;
    GAM1[eq_6, Epi_t] =1;
    PPI[eq_6, epi_sh] = 1;
#Ex_t

    GAM0[eq_7,x_t]   = 1;
    GAM1[eq_7, Ex_t] =1;
    PPI[eq_7, ex_sh] = 1;
#Ex_t2

    GAM0[eq_8,Ex_t]   = 1;
    GAM1[eq_8, Ex_t2] =1;
    PPI[eq_8, ex2_sh] = 1;
#Ei_t

    GAM0[eq_9,i_t]   = 1;
    GAM1[eq_9, Ei_t] =1;
    PPI[eq_9, ei_sh] = 1;   
    
#Ei_t2

    GAM0[eq_10,Ei_t]   = 1;
    GAM1[eq_10, Ei_t2] =1;
    PPI[eq_10, ei2_sh] = 1;    
#Epi_t2

    GAM0[eq_11,Epi_t]   = 1;
    GAM1[eq_11, Epi_t2] =1;
    PPI[eq_11, epi2_sh] = 1;    
    
    
    
    G1,impact,RC  = gensys(GAM0, GAM1, PSI, PPI, DIV=1 + 1e-8,REALSMALL=1e-6,return_everything=False)
    
    # GAM0*x(t) = A*x(t-1) + B*Ex(t+1) + C*Ex(t+2) + E*u(t) + D*eps(t)

    GAM0 = cp.array([[1,0,w2],[-n1,1,0],[-(1-rho)*phi_x,-(1-rho)*phi_pi,1]])
    GAM0inv = cp.linalg.inv(GAM0)
    A  = cp.matmul(GAM0inv,cp.array([[w4,0,0],[n2,n5,0],[0,0,rho]]))
    B = cp.matmul(GAM0inv,cp.array([[w1,w2,0],[n3,n4,0],[0,0,0]]))
    C = cp.matmul(GAM0inv,cp.array([[w3,0,0],[0,0,0],[0,0,0]]))
    D = cp.matmul(GAM0inv,cp.array(([0,0,0],[0,0,0],[0,0,1])))  
    E = cp.matmul(GAM0inv,cp.array([[-w2,0,0],[0,1,0],[0,0,1]]))
    R = cp.array([[para[0],0,0],[0,para[14],0],[0,0,para[1]]]) 
    # x(t) = A*x(t-1) + B*Ex(t+1) + C*Ex(t+2) + E*u(t) 
    V_s = cp.array([[para[8],0,0],[0,para[16],0],[0,0,para[15]]])
    
    D = cp.hstack([E,D])
    return G1,impact,RC,A,B,C,D,R,V_s,E


# In[7]:


para1 = cp.load(r'C:\Research\KF Results\no proj\POSTdist.npy')[10000:95000]
para =cp.zeros([18])
para[17] = .5
para[0:17] = cp.mean(para1,0)


# In[8]:


G1,impact,RC,A,B,C,D,R,V_s,E = REE_gen(para)


# In[9]:


R[2,2] = para[17]


Us = cp.zeros([3,21])
Rs = cp.zeros([3,21])
Q = cp.zeros([3,21])
Lx = cp.zeros([3,21])
Lpi = cp.zeros([3,21])
Li = cp.zeros([3,21])
I  = cp.zeros([3,21])
I[0,0] = 1
I[1,1] = 1
I[2,2] = 1

Lx[0,3] = 1
Lx[1,6] = 1
Lx[2,9] = 1

Lpi[0,4] =  1
Lpi[1,7] =  1
Lpi[2,10] = 1 


Li[0,5] =  1
Li[1,8] =  1
Li[2,11] = 1

Rs[0,12] = 1
Rs[1,15] = 1
Rs[2,18] =1

Us[0,13] = 1
Us[1,16] =1
Us[2,19] =1

Q[0,14] = 1
Q[1,17] =1
Q[2,20] =1





# In[10]:


PHI_og = .001*cp.random.standard_normal([21,1])
RR_og = 10*cp.random.standard_normal([21,21])


# In[51]:



DROP = 0
T = 150
Z = cp.zeros([3,T])
U = cp.zeros([3,T])
PHI = PHI_og
RR = RR_og
EZ = cp.zeros([3,1])
EZ1= EZ
EZ2 = EZ
V_s
PHI_old = PHI
Z_pred = cp.zeros([3,T])
for t in range(T):

    shock = m(V_s,cp.random.standard_normal([3,1]))
    U[:,t] = resh((resh(m(R,U[:,t-1]),[3,1])+shock),[3])
    
    PHI0  =  resh(PHI[0:3],[3,1])
    PHI1  =  resh(PHI[3:12],[3,3])
    PHI2  =  resh(PHI[12:],[3,3])
    
    EZ = PHI0+m(PHI1,resh(Z[:,t-1],[3,1]))+ m(PHI2,resh(U[:,t],[3,1]))
    if EZ[2]<0:
        EZ[2] = 0
    Z_pred[:,t] = resh(EZ,[3])
    EZ1 = PHI0+m(PHI1,resh(Z[:,t-1],[3,1]))+ m(PHI2,resh(U[:,t],[3,1]))
    if EZ1[2]<0:
        EZ1[2] = 0
    EZ2 = PHI0+m(PHI1,resh(Z[:,t-1],[3,1]))+ m(PHI2,resh(U[:,t],[3,1]))
    if EZ2[2]<0:
        EZ2[2] = 0
    Z[:,t] = resh((resh(m(A,Z[:,t-1]),[3,1])+m(B,EZ)+m(C,EZ2)+resh(m(E,U[:,t]),[3,1])),[3])
    if Z[2,t] < 0:
        Z[2,t] = 0 

    X = Lx*Z[0,t-1]+Lpi*Z[1,t-1]+Li*Z[2,t-1]+Rs*U[0,t-1]+Us*U[1,t-1]+Q*U[2,t-1] 
    
    err = resh(Z[:,t],[3,1])-resh(EZ,[3,1])
    RR = RR+.02*(m(X.T,X)-RR)
    PHI = PHI + .02*m(m(cp.linalg.inv(RR),X.T),err)
    if cp.any(abs(cp.linalg.eigvalsh(PHI1))>=1) == True:
                                                PHI = PHI_old
                                                DROP = DROP+1
                
    FE = cp.sum((Z-Z_pred)**2)
    dropout = DROP
    #print(abs(cp.linalg.eigvalsh(PHI11)))
            

print(cp.mean(FE))
print(DROP)


# In[52]:


print(cp.mean(FE))
print(DROP)


# In[53]:


plt.plot(cp.asnumpy(Z[2,:]))


# In[145]:


DROP = 0
T = 150
Z = cp.zeros([3,T])
U = cp.zeros([3,T])
PHI = .001*cp.random.standard_normal([21,1])
RR = 10*cp.random.standard_normal([21,21])
EZ = cp.zeros([3,1])
EZ1= EZ
EZ2 = EZ
ERR = cp.zeros([T,1])
V_s
PHI_old = PHI
Z_pred = cp.zeros([3,T])
for t in range(T):
    if t ==0:
        var = 1
    shock = m(V_s,cp.random.standard_normal([3,1]))
    U[:,t] = resh((resh(m(R,U[:,t-1]),[3,1])+shock),[3])
    
    PHI0  =  resh(PHI[0:3],[3,1])
    PHI1  =  resh(PHI[3:12],[3,3])
    PHI2  =  resh(PHI[12:],[3,3])
    EZ = PHI0+m(PHI1,resh(Z[:,t-1],[3,1]))+ m(PHI2,resh(U[:,t],[3,1]))
    if EZ[2]<0:
        EZ[2] = 1*sc.stats.truncnorm.mean(0,math.inf,np.float(Z[2,t-1]),np.float(var)**2)
        
    Z_pred[:,t] = resh(EZ,[3])
    
    EZ1 = PHI0+m(PHI1,resh(Z[:,t-1],[3,1]))+ m(PHI2,resh(U[:,t],[3,1]))
    if EZ1[2]<0:
        EZ1[2] = 1*sc.stats.truncnorm.mean(0,math.inf,np.float(EZ[2]),np.float(var)**2)

    EZ2 = PHI0+m(PHI1,resh(Z[:,t-1],[3,1]))+ m(PHI2,resh(U[:,t],[3,1]))
    if EZ2[2]<0:
        EZ2[2] = 1*sc.stats.truncnorm.mean(0,math.inf,np.float(EZ1[2]),np.float(var)**2)
    

    Z[:,t] = resh((resh(m(A,Z[:,t-1]),[3,1])+m(B,EZ)+m(C,EZ2)+resh(m(E,U[:,t]),[3,1])),[3])
    
    
   

    if Z[2,t] < 0:
            Z[2,t] = 0 
    
    X = Lx*Z[0,t-1]+Lpi*Z[1,t-1]+Li*Z[2,t-1]+Rs*U[0,t-1]+Us*U[1,t-1]+Q*U[2,t-1] 
    
    err = resh(Z[:,t],[3,1])-resh(EZ,[3,1])
    ERR[t] = cp.sum(err**2)
    var = cp.mean(ERR[0:t])
    RR = RR+.02*(m(X.T,X)-RR)
    PHI = PHI + .02*m(m(cp.linalg.inv(RR),X.T),err)
    if cp.any(abs(cp.linalg.eigvalsh(PHI1))>=1) == True:
                                                PHI = PHI_old
                                                DROP = DROP+1
                
    FE = cp.sum((Z-Z_pred)**2)
    dropout = DROP
    #print(abs(cp.linalg.eigvalsh(PHI11)))
            
print(cp.mean(FE)) 
#print(DROP)


# In[149]:


print(cp.mean(FE))
print(DROP)
plt.plot(cp.asnumpy(Z[2,:]))


# In[173]:


plt.plot(cp.asnumpy(Z_pred[0,:]))


# In[ ]:



'''
#sc.stats.truncnorm.mean(abs(cp.float(EZ[2])),math.inf,0,1)
    DROP = 0 
    T = 500
    #Z = cp.zeros((3,T))
    #U = cp.zeros([3,T])
    PHI = .01*cp.random.standard_normal([21,1])
    RR = 10*cp.random.standard_normal([21,1])
    EZ = cp.zeros([3,1])
    EZ1= EZ
    EZ2 = EZ
    V_s
    PHI_old = PHI
    n = 0
    sig = 1
Z_pred = cp.zeros([3,T])
for t in range(T):
        n = t+3
        shock = m(V_s,cp.random.standard_normal([3,1]))
        U[:,t] = resh((resh(m(R,U[:,t-1]),[3,1])+shock),[3])
        PHI_old = PHI

        PHI0  =  resh(PHI[0:3],[3,1])
        PHI1  =  resh(PHI[3:12],[3,3])
        PHI2  =  resh(PHI[12:],[3,3])
        #sig = (1/(n-2))*(sig*(n-3)+(err[2]**2))
        if t ==0:
            sig = 1
    
        EZ = PHI0+m(PHI1,resh(Z[:,t-1],[3,1]))+ m(PHI2,resh(U[:,t],[3,1]))
        if EZ[2]<0:
            lamb_0 = sc.stats.truncnorm.mean(abs(cp.float(EZ[2])),math.inf,0,1)
            EZ[2] = EZ[2]+lamb_0 
            #EZ[2] = 0
        EZ1 = PHI0+m(PHI1,resh(EZ,[3,1]))+ m(PHI2,resh(U[:,t],[3,1]))    
        if EZ1[2]<0:
            lamb_1 =sc.stats.truncnorm.mean(abs(cp.float(EZ1[2])),math.inf,0,1)
            EZ1[2] = EZ1[2]+lamb_1 
            #EZ1[2] = 0

        EZ2 = PHI0+m(PHI1,resh(EZ1,[3,1]))+ m(PHI2,resh(U[:,t],[3,1]))
        if EZ2[2]<0:
            lamb_2 = sc.stats.truncnorm.mean(abs(cp.float(EZ2[2])),math.inf,0,1)
            EZ2[2] = EZ2[2]+lamb_2
            #EZ2[2] = 0
            
            
            
            
            
            
        
        Z_pred[:,t] = resh(EZ,[3]) 

        Z[:,t] = resh((resh(m(A,Z[:,t-1]),[3,1])+m(B,EZ1)+m(C,EZ2)+resh(m(E,U[:,t]),[3,1])),[3])
        
        if Z[2,t] < 0:
            Z[2,t] = 0 
    
        X = Lx*Z[0,t-1]+Lpi*Z[1,t-1]+Li*Z[2,t-1]+Rs*U[0,t-1]+Us*U[1,t-1]+Q*U[2,t-1] 
        err = resh(Z[:,t],[3,1])-resh(EZ,[3,1])
        RR = RR+.02*(m(X.T,X)-RR)
        PHI = PHI +.02*m(m(cp.linalg.inv(RR),X.T),err)
        if cp.any(abs(cp.linalg.eigvalsh(PHI1))>=1) == True:
                                                PHI = PHI_old
                                                DROP = DROP+1
    
    
FE = cp.sum((Z_pred - Z)**2)
            
print(cp.mean(FE)) 
print(DROP)
'''


# In[ ]:





# In[ ]:





# In[ ]:




