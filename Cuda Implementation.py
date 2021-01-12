#!/usr/bin/env python
# coding: utf-8

# In[120]:


import numpy as np
import pycuda
import pycuda.driver as drv
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from sympy import Matrix 
import os
import time
import cupy as cp
import pandas as pd
if os.system("cl.exe"):
    os.environ['PATH'] += ';'+r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.28.29333\bin\Hostx64\x64"
if os.system("cl.exe"):
    raise RuntimeError("cl.exe still not found, path probably incorrect")
norm = np.random.standard_normal
resh = cp.reshape
y_data = np.array(pd.read_csv(r"C:\Research\ZLB stuff\New folder\DATA.csv"))




# In[121]:


mod= SourceModule("""
__global__ void mult( float *c,  float *b, float *a){
   

        int tx = threadIdx.x;
        
        c[tx] =a[tx] * b[tx];
  
}  
""")


# In[122]:


mod3= SourceModule("""
__global__ void matmult( const float *A, const float *B, float *C){
   
  int n = 10;
  
  int row= blockIdx.y*blockDim.y + threadIdx.y;
  int col = blockIdx.x*blockDim.x + threadIdx.x;

  if(row < n && col < n){
    float val = 0.0;
    for(int i=0; i<n; ++i){
      val  += A[row*n + i]*B[n*i + col];
    }
  C[row*n + col] = val;
  }
}  



  
  """)


# In[123]:


mod4= SourceModule("""
__global__ void matadd(float *A, float *B, float *C){
   
  int n = 10;
  
  int tx = blockIdx.y*blockDim.y + threadIdx.y;
  int ty = blockIdx.x*blockDim.x + threadIdx.x;
  int idx = ty*n+tx;
  if((ty <n) && (tx < n)){
      C[idx] = A[idx]+B[idx];
  }
  
  }
  """)


# In[124]:


n_x = 10 
n_y = 10
q = 1
w = 1
e = 1
GRID = (q,w,e)
BLOCK = (10,10,1)


# In[125]:


a = np.random.randn(n_x,n_y)*100
a = a.astype(np.float32)
b = np.random.randn(n_x,n_y)*100
b = b.astype(np.float32)
c = np.empty([n_x,n_y])
c = c.astype(np.float32)
a_gpu = cuda.mem_alloc(a.nbytes)
b_gpu = cuda.mem_alloc(b.nbytes)
c_gpu = cuda.mem_alloc(c.nbytes)
cuda.memcpy_htod(a_gpu, a)
cuda.memcpy_htod(b_gpu, b)
#matadd = mod4.get_function("matadd");
#matadd(a_gpu, b_gpu,c_gpu, block=BLOCK, grid=GRID);
matmult = mod3.get_function("matmult");
matmult(a_gpu, b_gpu,c_gpu, block=BLOCK, grid=GRID);
cuda.memcpy_dtoh(c, c_gpu)
#c-(a+b)
c-np.matmul(a,b)


# In[126]:


def Generate_Matrices(P):
    Thetac = cp.random.standard_normal([6,P])
    A_Q = cp.zeros([3,3,P])
    B_Q = cp.zeros([3,3,P])
    for i in range(P):
        k = Thetac[0,i] 
        sigma = Thetac[1,i] 
        phi_pi =Thetac[2,i]
        phi_x =Thetac[3,i] 
        Beta =Thetac[4,i]
        g =Thetac[5,i] 
        A = cp.array([[1,float(sigma),0],[0,float(Beta),0],[0,0,0]])
        B = cp.array([[float(sigma),0,0],[0,1,0],[0,0,1]])
        GAM0 = cp.array([[1,0,float(sigma)],[-float(k),1,0],[-float(phi_x),-float(phi_pi),1]])
        A = cp.matmul(np.linalg.inv(GAM0),A)
        B = cp.matmul(np.linalg.inv(GAM0),B)
        A_Q[:,:,i] = A  
        B_Q[:,:,i] = B
    A_Q  = cp.asnumpy(A_Q)
    B_Q  = cp.asnumpy(B_Q)
    return A_Q,B_Q


# In[ ]:





# In[127]:


mod= SourceModule("""

//#include <curand_kernel.h>
//#include <cuda_runtime.h>
//#include <cublas_v2.h>

__global__ void Invert(float* a){
//int lda = 10000;
//int *P; 
//int *INFO;
//int batchSize = 10;
//cublasSgetrfBatched(handle, n, RR, lda, P, INFO, batchSize)

int tx = blockIdx.x * blockDim.x + threadIdx.x;
//int ty = blockIdx.y * blockDim.y + threadIdx.y;
a[threadIdx.x] = 1;
}""",no_extern_c= False)


# In[128]:


mod = SourceModule("""
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
extern "C"{

__global__ void Invert(float *a_i, float *c_o, int n)
{
  int *p = (int *)malloc(3*sizeof(int));
  int *info = (int *)malloc(sizeof(int));
  int batch;
  cublasHandle_t hdl;
  info[0] = 0;
  batch = 1;
  float **a = (float **)malloc(sizeof(float *));
  *a = a_i;
  const float **aconst = (const float **)a;
  float **c = (float **)malloc(sizeof(float *));
  *c = c_o;
  
  //cublasSgetrfBatched(hdl, n, a, n, p, info, batch);
  
  }
}""",no_extern_c= True)


# In[250]:



mod = SourceModule("""
__global__ void Invert1(float *a_d , float *b_d) 
{
    int idx = threadIdx.x+blockIdx.x*blockDim.x ; 
    int idy = threadIdx.y+blockIdx.y*blockDim.y ; 
    a_d[idx] = 0;
}"""
 )


# In[257]:



mod = SourceModule("""
__global__ void Invert(float *a_d , float *b_d) 
{
    int idx = threadIdx.x +blockDim.X*block; 
    int idy = threadIdx.y ; 
    int size = 4;
    
    //Allocating memory in the share memory of the device 
    __shared__ float temp[4][4]; 
    
    //Copying the data to the shared memory 
    temp[idy][idx] = a_d[(idy * (size+1)) + idx] ; 
    
    for(int i =1 ; i<size ;i++) 
    { 
        if((idy + i) < size) // NO Thread divergence here 
        { 
            float var1 =(-1)*( temp[i-1][i-1]/temp[i+idy][i-1]); 
            temp[i+idy][idx] = temp[i-1][idx] +((var1) * (temp[i+idy ][idx]));
        } 
        __syncthreads(); //Synchronizing all threads before Next iterat ion 
    } 
    
    b_d[idy*(size+1) + idx] = temp[idy][idx]; 
}"""
 )


# In[267]:


n_R = 4
J = 4
P = 1
# Create Host Memory
RR =norm([n_R,n_R])
Rinv = RR
c = np.empty([n_R,n_R])
RR = RR.astype(np.float32)
Rinv = Rinv.astype(np.float32)
c = c.astype(np.float32)
# Create Device Memory
RR_gpu = cuda.mem_alloc(RR.nbytes)
Rinv_gpu = cuda.mem_alloc(Rinv.nbytes)
c_gpu = cuda.mem_alloc(c.nbytes)
INFO_gpu = cuda.mem_alloc(INFO.nbytes)
# Transfer host to device Data
cuda.memcpy_htod(RR_gpu, RR)
cuda.memcpy_htod(c_gpu, c)
cuda.memcpy_htod(Rinv_gpu, Rinv)
Invert = mod.get_function("Invert");
GRID = (1,1,1)
BLOCK = (1,1,1)
Invert(Rinv_gpu,c_gpu,block=BLOCK, grid=GRID);
#cuda.memcpy_dtoh(Rinv, Rinv_gpu)
cuda.memcpy_dtoh(c, c_gpu)

c  - np.linalg.inv(Rinv)


# In[263]:


np.linalg.inv(Rinv)


# In[221]:


c


# In[ ]:





# In[41]:


mod= SourceModule("""


__global__ void mult(float *U, float *EPS,float *X,float *C,float *Theta,float *Theta2,float *EX,float *EX1,float *PHI,float *RR,float *A_Q,float *B_Q){
     int tx = (blockIdx.x*blockDim.x + threadIdx.x);
 // Propogate AR
     int J = 10000;
     int P = 30;
     
   
     U[tx] =Theta[threadIdx.x]*U[tx]+Theta2[threadIdx.x]*EPS[tx];
     U[tx+J*P] =Theta[threadIdx.x+P]*U[tx+J*P]+Theta2[threadIdx.x+P]*EPS[tx+J*P];
     U[tx+J*P*2] =Theta[threadIdx.x+2*P]*U[tx+J*P*2]+Theta2[threadIdx.x+2*P]*EPS[tx+J*P*2];
    // Generate Expectations
      EX[tx] = PHI[tx]+U[tx]*PHI[tx+6*3]+U[tx+J*P]*PHI[tx+6*4]+U[tx+2*J*P]*PHI[tx+6*5];
      EX[tx+J*P] = PHI[tx+6*1]+U[tx]*PHI[tx+6*6]+U[tx+J*P]*PHI[tx+6*7]+U[tx+2*J*P]*PHI[tx+6*8];
      EX[tx+2*J*P] = PHI[tx+6*2]+U[tx]*PHI[tx+6*9]+U[tx+J*P]*PHI[tx+6*10]+U[tx+2*J*P]*PHI[tx+6*11];


      EX1[tx] = PHI[tx]+Theta[threadIdx.x]*U[tx]*PHI[tx+6*3]+Theta[threadIdx.x+P]*U[tx+J*P]*PHI[tx+6*4]+Theta[threadIdx.x+2*P]*U[tx+2*J*P]*PHI[tx+6*5];
      EX1[tx+J*P] = PHI[tx+6*1]+Theta[threadIdx.x]*U[tx]*PHI[tx+6*6]+Theta[threadIdx.x+P]*U[tx+J*P]*PHI[tx+6*7]+Theta[threadIdx.x+2*P]*U[tx+2*J*P]*PHI[tx+6*8];
      EX1[tx+2*J*P] = PHI[tx+6*2]+Theta[threadIdx.x]*U[tx]*PHI[tx+6*9]+Theta[threadIdx.x+P]*U[tx+J*P]*PHI[tx+6*10]+Theta[threadIdx.x+2*P]*U[tx+2*J*P]*PHI[tx+6*11];

    // Computing ALM

     X[tx] = A_Q[threadIdx.x]*EX1[tx]+A_Q[threadIdx.x+3]*EX1[tx+J*P];
     X[tx+J*P] = A_Q[threadIdx.x+3*3]*EX1[tx]+A_Q[threadIdx.x+3*4]*EX1[tx+J*P];
     X[tx+2*J*P] = A_Q[threadIdx.x+3*6]*EX1[tx]+A_Q[threadIdx.x+3*7]*EX1[tx+J*P];
     
     
    // Compute Matrix Inverse 
    RR[tx] = 0;
     
     
     
}
""")


# In[45]:


# Initialize Dimensions
n_x = 3;
n_phi = 12;
J = 10000
P =50
GRID = (J,1,1)
BLOCK = (P,n_x,1)
# Create Host Memory 
U = np.random.randn(n_x,J,P)
EPS = np.random.randn(n_x,J,P)
X = np.random.randn(n_x,J,P)
C = np.random.randn(n_x,J,P)
EX = np.zeros([n_x,J,P])
EX1 = EX
Theta = .01*np.random.randn(n_x,P)
Theta2 = .01*np.random.randn(n_x,P)
PHI =  np.random.standard_normal([n_phi,J,P])
A_Q,B_Q = Generate_Matrices(P) 
X =  np.random.standard_normal([n_x,J,P])
RR = np.random.randn(n_phi,n_phi,J,P)
w = np.random.randn(J,P)
# Convert Data type
U = U.astype(np.float32)
EPS = EPS.astype(np.float32)
C = C.astype(np.float32)
X = X.astype(np.float32)
RR = RR.astype(np.float32)
Theta = Theta.astype(np.float32)
Theta2 = Theta2.astype(np.float32)
EX = EX.astype(np.float32)
EX1 = EX1.astype(np.float32)
PHI = PHI.astype(np.float32)
A_Q = A_Q.astype(np.float32)
B_Q = B_Q.astype(np.float32)
w = w.astype(np.float32)
y_data = y_data.astype(np.float32)
Y =  y_data.astype(np.float32)
# Create Device Memory
U_gpu = cuda.mem_alloc(U.nbytes)
EPS_gpu = cuda.mem_alloc(EPS.nbytes)
X_gpu = cuda.mem_alloc(X.nbytes)
RR_gpu = cuda.mem_alloc(RR.nbytes)
Theta_gpu = cuda.mem_alloc(Theta.nbytes)
Theta2_gpu = cuda.mem_alloc(Theta.nbytes)
A_Q_gpu =  cuda.mem_alloc(A_Q.nbytes)
B_Q_gpu =  cuda.mem_alloc(A_Q.nbytes)
EX_gpu = cuda.mem_alloc(EX.nbytes)
EX1_gpu = cuda.mem_alloc(EX1.nbytes)
PHI_gpu = cuda.mem_alloc(PHI.nbytes)
C_gpu = cuda.mem_alloc(C.nbytes)
w_gpu = cuda.mem_alloc(w.nbytes)
y_data_gpu = cuda.mem_alloc(y_data.nbytes)

# Copy Host Memory to GPU
#c_gpu = cuda.mem_alloc(c.nbytes)
cuda.memcpy_htod(X_gpu, X)
cuda.memcpy_htod(RR_gpu, RR)
cuda.memcpy_htod(U_gpu, U)
cuda.memcpy_htod(EPS_gpu, EPS)
cuda.memcpy_htod(Theta_gpu, Theta)
cuda.memcpy_htod(Theta2_gpu, Theta2)
cuda.memcpy_htod(A_Q_gpu, A_Q)
cuda.memcpy_htod(B_Q_gpu, B_Q)
cuda.memcpy_htod(EX_gpu, EX)
cuda.memcpy_htod(EX1_gpu, EX)
cuda.memcpy_htod(PHI_gpu, PHI)
cuda.memcpy_htod(C_gpu, C)
cuda.memcpy_htod(w_gpu, w)
cuda.memcpy_htod(y_data_gpu, y_data)


mult = mod.get_function("mult");
mult(U_gpu,EPS_gpu,X_gpu,C_gpu,Theta_gpu,Theta2_gpu,EX_gpu,EX1_gpu,PHI_gpu,A_Q_gpu,B_Q_gpu,block=BLOCK, grid=GRID);
# Return Values from Fxn
cuda.memcpy_dtoh(U, U_gpu)
cuda.memcpy_dtoh(X, X_gpu)
cuda.memcpy_dtoh(C, C_gpu)
cuda.memcpy_dtoh(Theta, Theta_gpu)
cuda.memcpy_dtoh(A_Q, A_Q_gpu)
cuda.memcpy_dtoh(PHI, PHI_gpu)
cuda.memcpy_dtoh(EX, EX_gpu)
cuda.memcpy_dtoh(EX1, EX1_gpu)
cuda.memcpy_dtoh(w, w_gpu)
cuda.memcpy_dtoh(Y, y_data_gpu)
cuda.memcpy_dtoh(RR, RR_gpu)



# In[47]:


RR[0,0,0]


# In[48]:


import time
start = time.time()
stop = time.time()
stop - start


# In[69]:


RR = cp.asnumpy(resh(cp.linalg.inv(cp.array(RR.T)),[12,12,J,P]))


# In[72]:


start = time.time()
X_b = norm([J,P])
for t in range(500):
    mult(U_gpu,EPS_gpu,X_gpu,C_gpu,Theta_gpu,Theta2_gpu,EX_gpu,EX1_gpu,PHI_gpu,A_Q_gpu,B_Q_gpu,w_gpu,block=BLOCK, grid=GRID);
    cuda.memcpy_dtoh(X, X_gpu)
    cuda.memcpy_dtoh(U, U_gpu)
    w = (y_data[t,0]- (X[0]-X_b + U[0]))**2+(y_data[t,1]- (X[1]))**2+(y_data[t,2]- (X[2]))**2
    X_b = X[0]
stop = time.time()
duration = stop - start
print(duration/P)


# In[43]:


w = np.exp(-.5*(y_data[t,0]- (X[0]-X_b + U[0]))+(y_data[t,1]- (X[1]))+(y_data[t,2]- (X[2])))
idx = np.random.choice(np.arange(J), J, replace=True,p=cp.asnumpy(w))


# In[ ]:




