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
from datetime import datetime
import seaborn as sns
import math
import time
ordqz = sc.linalg.ordqz
svd = sc.linalg.svd
import pandas as pd
resh  = cp.reshape
m  = cp.matmul
norm = cp.random.standard_normal
zeros = cp.zeros


# In[2]:


y_data = cp.array(pd.read_csv(r"C:\Research\ZLB stuff\New folder\DATA.csv"))
FFR = np.array(pd.read_csv(r"C:\Research\ZLB stuff\New folder\fedfundz.csv"))


# In[3]:


pd.read_csv(r"C:\Research\ZLB stuff\New folder\DATA.csv").head()


# In[4]:


T =533
para = cp.zeros([10,1])
LL = cp.zeros([T,1])
ffr = cp.zeros([T,1])
P_1yr = cp.zeros([T,1]) 
output = cp.zeros([T,1])
EZ = cp.random.standard_normal([3,1])
S = 1*cp.random.standard_normal([12,1])
PHI = .01*cp.random.standard_normal([12,1])
RR =100*cp.random.standard_normal([12,12])
P = cp.random.standard_normal([12,12])
P0 = .1*cp.random.standard_normal([12,12])
Rinv = 10*cp.zeros([12,12])
k = .2
p_r = .5
sigma = .125
phi_pi = 1.5
phi_x = .5
Beta = .99
g = .02
sig_r = 1.0
p_u = .1
p_q = 0.5
sig_u = 1.0
sig_q = 1.0

para[0] = p_r
para[1] = sigma
para[2] = phi_pi
para[3] = phi_x
para[4] = Beta
para[5] = g
para[6] = sig_r
para[7] = sig_u
para[8] = sig_q
para[9] = p_q


#G1 = REE(para)
#PHI[3:] = cp.array(resh(G1[0:3,3:6],[9,1]))
#PHI[0:3] =0*PHI[0:3]
Xb = S[0]
x0 = .01*cp.random.standard_normal([12,1])
error = cp.zeros(T)
y_pred = cp.zeros([3,T])
A = cp.array([[1,float(sigma),0],[0,float(Beta),0],[0,0,0]])
B = cp.array([[float(sigma),0,0],[0,1,0],[0,0,1]])
GAM0 = cp.array([[1,0,float(sigma)],[-float(k),1,0],[-float(phi_x),-float(phi_pi),1]])
A = cp.matmul(cp.linalg.inv(GAM0),A)
B = cp.matmul(cp.linalg.inv(GAM0),B)
M1 = cp.array([[.01,0,0,0,0,0],[0,.01,0,0,0,0],[0,0,.01,0,0,0]])
M2 = cp.array([[-.01,0,0,0,0,0],[0,-.01,0,0,0,0],[0,0,-.01,0,0,0]])
MM = cp.hstack((M1,M2))
V_S_bar = cp.diag([(sig_r**2)*(p_r**2),(sig_u**2)*(p_u**2),(p_q**2)*(sig_q**2),(sig_r**2),(sig_u**2),(sig_q**2)])
V_s = cp.hstack([cp.vstack([V_S_bar,cp.zeros([6,6])]),cp.zeros([12,6])])
V_m = 1*cp.eye(3)
Us = cp.vstack([cp.zeros([1,3]),cp.zeros([1,3]),cp.zeros([1,3]),cp.zeros([1,3]),[1,0,0],cp.zeros([1,3]),cp.zeros([1,3]),[0,1,0],cp.zeros([1,3]),cp.zeros([1,3]),[0,0,1],cp.zeros([1,3])])
Rs = cp.vstack([cp.zeros([1,3]),cp.zeros([1,3]),cp.zeros([1,3]),[1,0,0],cp.zeros([1,3]),cp.zeros([1,3]),[0,1,0],cp.zeros([1,3]),cp.zeros([1,3]),[0,0,1],cp.zeros([1,3]),cp.zeros([1,3])])
Qs = cp.vstack([cp.zeros([1,3]),cp.zeros([1,3]),cp.zeros([1,3]),cp.zeros([1,3]),cp.zeros([1,3]),[1,0,0],cp.zeros([1,3]),cp.zeros([1,3]),[0,1,0],cp.zeros([1,3]),cp.zeros([1,3]),[0,0,1]])
I = cp.vstack([[1,0,0],[0,1,0],[0,0,1],cp.zeros([1,3]),cp.zeros([1,3]),cp.zeros([1,3]),cp.zeros([1,3]),cp.zeros([1,3]),cp.zeros([1,3]),cp.zeros([1,3]),cp.zeros([1,3]),cp.zeros([1,3])])
R = cp.array([[cp.float(p_r),0,0],[0,cp.float(p_u),0],[0,0,cp.float(p_q)]])
PHI_old = PHI
drop = 0
for t in range(T):
    PHI0 = PHI[0:3]
    PHI1 = cp.reshape(PHI[3:],[3,3])
    L_1 =  cp.matmul(cp.matmul(A,PHI1),R)+cp.matmul(B,R)
    L_0 =   cp.vstack([PHI0,0,0,0])
    L0 = cp.vstack([L_0,cp.zeros([6,1])])
    Q = cp.hstack([cp.zeros([6,3]),cp.vstack([L_1,R])])
    L1 = cp.hstack([cp.vstack((Q,cp.identity(6))),(cp.zeros([12,6]))])
    x = L0 + cp.matmul(L1,x0)
    y = cp.matmul(MM,x)
    P = cp.matmul(cp.matmul(L1,P0),L1.T) + V_s
    D = cp.matmul(cp.matmul(MM,P0),MM.T) + V_m
    L = cp.matmul(P,MM.T)
    x0 = x + cp.matmul(cp.matmul(L,cp.linalg.inv(D)),(cp.reshape(y_data[t,0:3],[3,1])-y))
    if x0[2]<0:
        x0[2] = 0
    P0 = P - cp.matmul(cp.matmul(L,cp.linalg.inv(D)),L.T)
    EX = PHI0+m(PHI1,resh(x0[3:6],[3,1]))
    if EX[2]<0:
        EX[2] = 0
    EX1 = PHI0+m(PHI1,m(R,resh(x0[3:6],[3,1])))
    #EX2 = PHI0+m(PHI1,m(R**2,resh(x0[3:6],[3,1])))
    V = m(m(m(PHI1,R),V_S_bar[0:3,0:3]),m(PHI1,R).T)
    Var = (sigma**2)*V[0,0]-(2*sigma)*V[0,1]-V[1,1]
    Z  =cp.exp(sigma*(EX1[0]-x0[0])-(EX1[1]-.5*Var))
    P_1yr[t] = Z    
    
    
    y_pred[:,t] = resh(cp.matmul(MM,x0),[3])
    ffr[t] = resh(x0[2],[1])
    output[t] = resh(x0[0],[1])

    err = cp.reshape(y_data[t,0:3],[3,1]) - cp.matmul(MM,x0)
    LL[t] = ((2*math.pi)**(-3/2))*cp.exp((-1/2)*(err[0]**2+err[1]**2+err[2]**2))

    error[t] = cp.sum(error)

    
    Xt = (x0[4]*Us) +(x0[3]*Rs) +I+(x0[5]*Qs)
    RR = (1-g)*RR + g*cp.matmul(Xt,Xt.T)
    ERR = cp.reshape(EX - x0[0:3],[3,1])
    PHI_old = PHI
    PHI = PHI + g*cp.matmul(cp.matmul(cp.linalg.inv(RR.T).T,Xt),ERR)

    
            
likic = cp.log(cp.sum(LL))
    
    
    
    

print(likic)    
plt.plot(cp.asnumpy(ffr))
print(drop)


# In[5]:


plt.plot(cp.asnumpy(P_1yr**1))


# In[156]:


################################ Particle Filter #################################################


# In[85]:


# Set up Parameters
para = cp.zeros([11])
k = .1
p_r = .5
sigma = .125
phi_pi = 1.5
phi_x = .5
Beta = .99
g = .025
sig_r = 2.6
p_u = .3
p_q = 0.2
sig_u = 0.2
sig_q = 0.1
# Find the Rational Expectations EQ
para[0] = p_r
para[1] = sigma
para[2] = phi_pi
para[3] = phi_x
para[4] = Beta
para[5] = g
para[6] = sig_r
para[7] = sig_u
para[8] = sig_q
para[9] = p_q
para[10] = p_u
#G1 = REE(para)
#G1 = cp.vstack([cp.zeros([3,1]),cp.array(resh(G1[0:3,3:6],[9,1]))])


# In[ ]:


sig


# In[207]:


SIG = cp.array([[sig_r,0,0],[0,sig_u,0],[0,0,sig_q]])
UU=  cp.random.multivariate_normal(cp.zeros([3]),SIG,[10000]).T
#UU = cp.random.uniform(0,1,[3,J])


# In[208]:


c1 = ((math.pi*2)**(3)*sig_q*sig_u*sig_q)**-(3/2)
PDF =c1*cp.exp(-.5*((UU[0,:]**2)*(sig_r**-2)+(UU[1,:]**2)*(sig_u**-2)+(UU[2,:]**2)*(sig_q**-2)))
SS = ((P[0]*UU[0,:]+P[1]*UU[1,:]+P[2]*UU[2,:]))*cp.exp((P[0]*UU[0,:]+P[1]*UU[1,:]+P[2]*UU[2,:])-.5*PDF)


# In[209]:


cp.mean(cp.asnumpy((SS*PDF)))


# In[184]:


cp.mean(cp.exp((P[0]*UU[0]+P[1]*UU[1]+P[2]*UU[2])-.5*PDF))


# In[185]:


cp.mean(((P[0]*UU[0,:]+P[1]*UU[1,:]+P[2]*UU[2,:])))


# In[187]:


PDF


# In[ ]:





# In[ ]:





# In[86]:


J = 10000
# Generate Macroeconomic Variables
EPS = norm([3,T,J])
PHI = .01*norm([12,J])
X_B = .01*norm([3,J])
Z_B = .01*norm([3,J])
Z =  .1*norm([3,J])
RR = 70*norm([12,12,J])
EX = norm([3,J])
EX1 = norm([3,J])
EX2 = norm([3,J])
X = .01*norm([3,J])
U = 1*norm([3,J])
P = 1*norm([3,J])
S = norm([5,T])
U[0,:] =sig_r*U[0,:]
U[1,:] =sig_u*U[1,:]
U[2,:] =sig_q*U[2,:]
A = cp.array([[1,float(sigma),0],[0,float(Beta),0],[0,0,0]])
B = cp.array([[float(sigma),0,0],[0,1,0],[0,0,1]])
GAM0 = cp.array([[1,0,float(sigma)],[-float(k),1,0],[-float(phi_x),-float(phi_pi),1]])
A = cp.matmul(cp.linalg.inv(GAM0),A)
B = cp.matmul(cp.linalg.inv(GAM0),B)
PP = zeros([12,12,J])
v=  0
drop = 0
for t in range(T):
    # Generate Macroeconomic shocks

    U[0] = p_r*U[0]+sig_r*norm(J)
    U[1] = p_u*U[1]+sig_u*norm(J)
    U[2] = p_q*U[2]+sig_q*norm(J)
    

    
    # Generate Expectations 
    EX[0] = PHI[0] + PHI[3]*U[0]+PHI[4]*U[1]+PHI[5]*U[2]
    EX[1] = PHI[1] + PHI[6]*U[0]+PHI[7]*U[1]+PHI[8]*U[2]
    EX[2] = PHI[2] + PHI[9]*U[0]+PHI[10]*U[1]+PHI[11]*U[2]
    
    EX[2,cp.where(EX[2]<0)[0]] =0 

    EX1[0] = PHI[0] + p_r*PHI[3]*U[0]+p_u*PHI[4]*U[1]+p_q*PHI[5]*U[2]
    EX1[1] = PHI[1] + p_r*PHI[6]*U[0]+p_u*PHI[7]*U[1]+p_q*PHI[8]*U[2]
    EX1[2] = PHI[2] + p_r*PHI[9]*U[0]+p_u*PHI[10]*U[1]+p_q*PHI[11]*U[2]
    
    EX1[2,cp.where(EX1[2]<0)[0]] =0

    
    # Generate Macroeconomic Variables
    X[0] = cp.sum((resh(A[0,:],[3,1])*EX1),0)+cp.sum((resh(B[0,:],[3,1])*U),0)
    X[1] = cp.sum((resh(A[1,:],[3,1])*EX1),0)+cp.sum((resh(B[1,:],[3,1])*U),0)
    X[2] = cp.sum((resh(A[2,:],[3,1])*EX1),0)+cp.sum((resh(B[2,:],[3,1])*U),0)
    X[2,cp.where(X[2]<0)[0]] = 0
    
    # Generate Financial Asset Prices
    var_x = (PHI[3]**2)*(sig_r**2)+(PHI[4]**2)*(sig_u**2)+(PHI[5]**2)*(sig_q**2)
    cov_xpi = (PHI[3]*PHI[6])*(sig_r**2)+(PHI[4]*PHI[7])*(sig_u**2)+(PHI[5]*PHI[8])*(sig_q**2)
    var_pi = (PHI[6]**2)*(sig_r**2)+(PHI[7]**2)*(sig_u**2)+(PHI[8]**2)*(sig_q**2)


    V1 = (sigma**2)*var_x-(2*sigma)*cov_xpi+(sigma**2)*var_pi
    mu  =sigma*(EX1[0]-X[0])+EX1[1]
    # 1 yr 
    Z[0]  = cp.log((Beta*cp.exp((mu+.5*V1))))
    Z[0,cp.where(Z[0]<0)[0]] = 0



    # 2 yr 
    
    # Compute Variance
    P[0] = PHI[3]+ PHI[6]
    P[1] = PHI[4]+ PHI[7]
    P[2] = PHI[5]+ PHI[8]
    V1 = (P[0]**2)*sig_r+(P[1]**2)*sig_u+(P[2]**2)*sig_q
    cov = cp.exp(PHI[2])*(PHI[9]*p_r*U[0]*P[0])+(PHI[10]*p_u*U[1]*P[1])+(PHI[11]*p_q*U[2]*P[2])
    cov = (PHI[9]*p_r*U[0]*P[0])+(PHI[10]*p_u*U[1]*P[1])+(PHI[11]*p_q*U[2]*P[2])
    V2 = (sig_r*PHI[9])**2 + (sig_u*PHI[10])**2+(sig_q*PHI[11])**2
    V2 = V1-2*(cov)+V2 
    Z[1]  =cp.log(Beta*cp.exp((mu+cp.exp(EX1[2]))))
    Z[1,cp.where(Z[1]<0)[0]] = 0
    
    # SPY     


    
    # RLS
    ERR =   (X[0:3] - EX)
    
    
    eps = cp.vstack([        ERR[0],ERR[1],ERR[2],
                             U[0]*ERR[0],U[1]*ERR[0],U[2]*ERR[0],
                             U[0]*ERR[1],U[1]*ERR[1],U[2]*ERR[1],
                             U[0]*ERR[2],U[1]*ERR[2],U[2]*ERR[2]
                    
                    ])
    
    
    
    
    Q = cp.array([[U[0]*U[0],U[0]*U[1],U[0]*U[2]],[U[0]*U[1],U[1]*U[1],U[1]*U[2]],[U[0]*U[2],U[2]*U[1],U[2]*U[2]]])

    PP[0:3,0:3] = resh(cp.repeat(cp.eye(3),J),[3,3,J])
    PP[0,0] = 1
    PP[1,1] = 1
    PP[2,2] = 1
    PP[0,3:6] = U
    PP[1,6:9] = U
    PP[2,9:] = U

    PP[3:6,0] = U
    PP[6:9,1] = U
    PP[9:,2] = U


    PP[3:6,3:6] = Q
    PP[6:9,6:9] = Q
    PP[9:,9:] = Q
    
    
    RR = RR+g*(RR-PP)
    Rinv = cp.linalg.inv(RR.T).T
    PHI = PHI+g*cp.sum(cp.reshape(cp.array(list(eps)*12),[12,12,J])*Rinv,1) 
    

    w = (((2*math.pi)**(-5/2)))*cp.exp(-.5*(
        
            (cp.float(y_data[t,0])-.1*(X[0]-X_B[0]))**2
            +(cp.float(y_data[t,1])-.1*(X[1]-X_B[1]))**2
            +(cp.float(y_data[t,2])-.1*(X[2]-X_B[2]))**2
            +(cp.float(y_data[t,3])-.1*(Z[0]-Z_B[0]))**2
            +(cp.float(y_data[t,4])-.1*(Z[1]-Z_B[1]))**2
            # cp.float(y_data[t,5])-.01*(Z[2]-Z_B[2])**2))
                   )
              
              )

    
    LL[t] = cp.mean(w)
    w[cp.where(cp.isnan(w))[0]] = 0
    w[cp.where(cp.isinf(w))[0]] = 0
    w = w/cp.sum(w)
    
    try:
        idx = np.random.choice(np.arange(J), J, replace=True,p=cp.asnumpy(w))
        X = X[:,idx]
        PHI = PHI[:,idx]
        RR = RR[:,:,idx]
        U = U[:,idx]
        Z = Z[:,idx]
        v = v+1
    except: 
        pass
    
    
    # GDP 
    S[0,t] = cp.mean(X[0]-X_B[0])
    # PI 
    S[1,t] = cp.mean(X[1]-X_B[1])
    # FFR
    S[2,t] = cp.mean(X[2])
    # 1 yr yield
    S[3,t] = cp.mean(Z[0])
    # 2 yr yield
    S[4,t] = cp.mean(Z[1])
    # 3 yr yield
    #S[5,t] = cp.mean(Z[2])

                
    X_B = X
    Z_B = Z
likic = cp.log(cp.sum(LL))
print(likic)
print(cp.where(cp.isnan(LL))[0])



fig1, ax1= plt.subplots()
ax1.plot(np.arange(T), cp.asnumpy(S[2]))
ax1.set_title("FFR")
ax1.set_xlabel("Time")
fig2, ax2 = plt.subplots() 
ax2.plot(np.arange(T),  cp.asnumpy(S[3]))
ax2.set_title("1 Year Return")
ax2.set_xlabel("Time")

fig3, ax3 = plt.subplots() 
ax3.plot(np.arange(T),  cp.asnumpy(S[4]))
ax3.set_title("2 Year Return")
ax3.set_xlabel("Time")

#fig3, ax3= plt.subplots()
#ax3.plot(np.arange(T),  cp.asnumpy(S[4]))
#ax3.set_title("2 Year Return")
#ax3.set_xlabel("Time")


# In[ ]:


### Estimation Phase #######


# In[82]:


cp.where(w<0)


# In[2]:


def PFLIKE(para):
    para[0] = p_r
    para[1] = sigma
    para[2] = phi_pi
    para[3] = phi_x
    para[4] = .995
    para[5] = g
    para[6] = sig_r
    para[7] = sig_u
    para[8] = sig_q
    para[9] = p_q
    para[10] = p_u
    J = 10000
    # Generate Macroeconomic Variables
    PHI =.1*norm([12,J])
    Z_B = norm([J])
    RR = 30*norm([12,12,J])
    X = norm([3,J])
    X_B = .1*norm([3,J])
    EX = norm([3,J])
    EX1 = norm([3,J])
    EX2 = norm([3,J])
    U = norm([3,J]) 
    U[0] =sig_r*norm([J])
    U[1] =sig_u*norm([J])
    U[2] =sig_q*norm([J])
    A = cp.array([[1,float(sigma),0],[0,float(Beta),0],[0,0,0]])
    B = cp.array([[float(sigma),0,0],[0,1,0],[0,0,1]])
    GAM0 = cp.array([[1,0,float(sigma)],[-float(k),1,0],[-float(phi_x),-float(phi_pi),1]])
    A = cp.matmul(cp.linalg.inv(GAM0),A)
    B = cp.matmul(cp.linalg.inv(GAM0),B)
    PP = zeros([12,12,J])
    v=  0
    for t in range(T):
    # Generate Macroeconomic shocks

        U[0] = p_r*U[0]+sig_r*norm(J)
        U[1] = p_u*U[1]+sig_u*norm(J)
        U[2] = p_q*U[2]+sig_q*norm(J)

    # Generate Expectations 
        EX[0] = PHI[0] + PHI[3]*U[0]+PHI[4]*U[1]+PHI[5]*U[2]
        EX[1] = PHI[1] + PHI[6]*U[0]+PHI[7]*U[1]+PHI[8]*U[2]
        EX[2] = PHI[2] + PHI[9]*U[0]+PHI[10]*U[1]+PHI[11]*U[2]
    
        EX[2,cp.where(EX[2]<0)[0]] =0 

        EX1[0] = PHI[0] + p_r*PHI[3]*U[0]+p_u*PHI[4]*U[1]+p_q*PHI[5]*U[2]
        EX1[1] = PHI[1] + p_r*PHI[6]*U[0]+p_u*PHI[7]*U[1]+p_q*PHI[8]*U[2]
        EX1[2] = PHI[2] + p_r*PHI[9]*U[0]+p_u*PHI[10]*U[1]+p_q*PHI[11]*U[2]
    
        EX1[2,cp.where(EX1[2]<0)[0]] =0

    
        # Generate Macroeconomic Variables
        X[0] = cp.sum((resh(A[0,:],[3,1])*EX1),0)+cp.sum((resh(B[0,:],[3,1])*U),0)
        X[1] = cp.sum((resh(A[1,:],[3,1])*EX1),0)+cp.sum((resh(B[1,:],[3,1])*U),0)
        X[2] = cp.sum((resh(A[2,:],[3,1])*EX1),0)+cp.sum((resh(B[2,:],[3,1])*U),0)
    
        X[2,cp.where(X[2]<0)[0]] = 0
    
        # Generate Financial Asset Prices
        var_x = (PHI[3]**2)*(p_r**2)*(sig_r**2)+(PHI[4]**2)*(p_u**2)*(sig_u**2)+(PHI[5]**2)*(p_q**2)*(sig_q**2)
        cov_xpi = (PHI[3]*PHI[6])*(p_r**2)*(sig_r**2)+(PHI[4]*PHI[7])*(p_u**2)*(sig_u**2)+(PHI[5]*PHI[8])*(p_q**2)*(sig_q**2)
        var_pi = (PHI[6]**2)*(p_r**2)*(sig_r**2)+(PHI[7]**2)*(p_u**2)*(sig_u**2)+(PHI[8]**2)*(p_q**2)*(sig_q**2)
        Var = (sigma**2)*var_x-(2*sigma)*cov_xpi-var_pi
        Z  =Beta*cp.exp(sigma*(EX1[0]-EX[0])-EX1[1]+.5*Var)

        # RLS
        ERR =   (X[0:3] - EX)
    
        eps = cp.vstack([        ERR[0],ERR[1],ERR[2],
                             U[0]*ERR[0],U[1]*ERR[0],U[2]*ERR[0],
                             U[0]*ERR[1],U[1]*ERR[1],U[2]*ERR[1],
                             U[0]*ERR[2],U[1]*ERR[2],U[2]*ERR[2]
                    
                    ])
    
    
    
        Q = cp.array([[U[0]*U[0],U[0]*U[1],U[0]*U[2]],[U[0]*U[1],U[1]*U[1],U[1]*U[2]],[U[0]*U[2],U[2]*U[1],U[2]*U[2]]])
    
        PP[0:3,0:3] = resh(cp.repeat(cp.eye(3),J),[3,3,J])
        PP[0,0] = 1
        PP[1,1] = 1
        PP[2,2] = 1
        PP[0,3:6] = U
        PP[1,6:9] = U
        PP[2,9:] = U

        PP[3:6,0] = U
        PP[6:9,1] = U
        PP[9:,2] = U


        PP[3:6,3:6] = Q
        PP[6:9,6:9] = Q
        PP[9:,9:] = Q

    
        RR = RR+g*(RR-PP)
        Rinv = cp.linalg.inv(RR.T).T
        PHI = PHI+g*cp.sum(cp.reshape(cp.array(list(eps)*12),[12,12,J])*Rinv,1)
    

        w = (((2*math.pi)**(-4/2)))*cp.exp(-.5*((cp.float(y_data[t,0])-.01*(X[0]-X_B[0]))**2+
            (cp.float(y_data[t,1])-.01*(X[1]-X_B[1]))**2+
            (cp.float(y_data[t,2])-.01*(X[2]-X_B[2]))**2+
            (cp.float(y_data[t,3])-.01*(Z-Z_B))**2

                   )
              
              )

    

        LL[t] = cp.mean(w)
        w[cp.where(cp.isnan(w))[0]] = 0
        w[cp.where(cp.isinf(w))[0]] = 0
        w = w/cp.sum(w)
    
        try:
            idx = np.random.choice(np.arange(J), J, replace=True,p=cp.asnumpy(w))
            X = X[:,idx]
            PHI = PHI[:,idx]
            RR = RR[:,:,idx]
            U = U[:,idx]
            Z = Z[idx]
            v = v+1
        except: 
            pass
        X_B = X
        Z_B = Z
    likic = cp.log(cp.sum(LL))


    return likic


# In[ ]:


def param_checks(para):
    para[0] = p_r
    para[1] = sigma
    para[2] = phi_pi
    para[3] = phi_x
    para[4] = .995
    para[5] = g
    para[6] = sig_r
    para[7] = sig_u
    para[8] = sig_q
    para[9] = p_q
    para[10] = p_u
    if cp.any(para<0)or p_r>1or p_u>1or p_q>1 or .01>g>.05 or phi_pi>2.2 or .09>phi_x>1 or sig_r>3 or sig_u>3 or sig_q>3 or sigma>1: 
        x = 0
    else: 
        x=1
    return x


# In[ ]:


param_checks(para)


# In[ ]:


# Begin MH
i = 0
Nsim = 100500
m = 1000
Thetasim = cp.zeros([Nsim,11])
logpost = cp.zeros([Nsim])
LIK = cp.zeros([Nsim])
AA = cp.zeros([Nsim])
DROP = cp.zeros([Nsim])
Thetasim[i,:] = resh(para,[11])
c = .02
P3 = cp.eye(11)
accept = 0
likij =  PFLIKE(Thetasim[i,:])
obj = cp.float(likij)
LIK[i] =cp.float(likij)
logpost[i] = obj
print('likelihood:', likij)
param_checks(para)


# In[ ]:


go_on = 0
v = 0
AR = 0
while i<Nsim:
    if i ==5000:
        P3 = cp.cov(Thetasim[0:i],rowvar= False)
    while go_on == 0:
            Thetac = cp.random.multivariate_normal(Thetasim[i,:],(c)*P3)
            go_on = param_checks(Thetac) 
    likic = PFLIKE(Thetac)
    objc =  cp.float(likic)
    if cp.isfinite(objc) == False:
        alpha = -1
        u = 0
    else:
        u = cp.log(cp.random.uniform(0,1,1))
        alpha = objc-obj
    if  alpha-u>=0:
        Thetasim[i+1,:] = Thetac
        accept            = accept+1;
        logpost[i+1] = objc
        LIK[i+1] = likic
        obj = objc
        likij = likic
        #print('accepted')
    else:
        Thetasim[i+1,:] = Thetasim[i,:]
        logpost[i+1] = obj
        LIK[i+1] = likij
    AR= (accept/(i+1))*100
    if v == m:
                    
                    v = 0
                    if  AR>90:
                                        c = c*2
                    if 90>AR and AR>75:
                                        c = c*1.6
                    if 75>AR and AR>50:
                                        c = c*1.5
                    if 50>AR and AR>45:
                                        c = c*1.3
                    if 45>AR and AR>40:
                                        c = c*1.26
                    if 40>AR and AR>35:
                                        c = c*1.25
                    if 35>AR and AR>33:
                                        c = c*1.05
                    if 33>AR and AR >30:
                                        c = c*1
                    if 20>AR and AR >15:
                                        c = c*.45
                    if 15>AR and AR>10:
                                        c = c*.03
                    if 10>AR and AR >5:
                                        c = c*.01
                    if 5>AR:
                            c = c*.001
    AA[i] = AR;
    if (i % 500) == 1:
        print("Iteration: ", i)
        print("Acceptance Rate: ",AR)
        print('Avg Likelihood:', cp.mean(LIK[0:i]))
        print("cov mult: ",c)
        print("p_r:",cp.mean(Thetasim[0:i,:],0)[0])
        print("sigma:",((cp.mean(Thetasim[0:i,:],0)[1])**-1))
        print("phi_pi:",cp.mean(Thetasim[0:i,:],0)[2])
        print("phi_x:",cp.mean(Thetasim[0:i,:],0)[3])
        print("Beta:",.99)
        print("g:",cp.mean(Thetasim[0:i,:],0)[5])
        print("sig_r:",cp.mean(Thetasim[0:i,:],0)[6])
        print("sig_u:",cp.mean(Thetasim[0:i,:],0)[7])
        print("sig_q:",cp.mean(Thetasim[0:i,:],0)[8])
        print("p_q:",cp.mean(Thetasim[0:i,:],0)[9])
        print("p_u:",cp.mean(Thetasim[0:i,:],0)[10])
        cp.save(r'C:\Research\ZLB stuff\Results\PostDIST', Thetasim)
        cp.save(r'C:\Research\ZLB stuff\Results\Acceptance', AA)
        cp.save(r'C:\Research\ZLB stuff\Results\logpost', logpost)
        cp.save(r'C:\Research\ZLB stuff\Results\likelyhood', LIK)
        cp.save(r'C:\Research\ZLB stuff\Results\covmat', P3)
        cp.save(r'C:\Research\ZLB stuff\Results\covmult', c)
        cp.save(r'C:\Research\ZLB stuff\Results\iter', i)
        print("Current Time =",datetime.now().strftime("%H:%M"))
    go_on = 0
    i = i+1
    v = v+1


# In[ ]:


i


# In[ ]:




