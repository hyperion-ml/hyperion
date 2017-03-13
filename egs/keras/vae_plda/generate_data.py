import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py
from scipy.stats import wishart
from six.moves import xrange

rng=np.random.RandomState(256)

D=100
Dy=2
Dz=50

M=150
N_i=8

VU=np.expand_dims(np.linspace(10,0.1,Dy+Dz),axis=-1)*rng.normal(size=(Dy+Dz,D)).astype(dtype='float32')
mu=np.zeros((1,D),dtype='float32')
x_var=rng.uniform(low=0.5,high=2,size=(1,D)).astype(dtype='float32')

[l, VU]=np.linalg.eigh(np.dot(VU.T,VU))
VU*=l
VU=VU[:,::-1].T

S=np.diag(np.dot(VU.T,VU)+np.diag(x_var))
VU/=np.sqrt(S)
x_var=x_var/S

V=VU[:Dy,:]
U=VU[Dy:Dy+Dz,:]

for s in xrange(3):

    y=rng.normal(size=(M,Dy)).astype(dtype='float32')
    z=rng.normal(size=(M*N_i,Dz)).astype(dtype='float32')
    epsilon=np.sqrt(x_var)*rng.normal(size=(M*N_i,D)).astype(dtype='float32')
    
    yy=np.repeat(y,N_i,axis=0)
    x=mu+np.dot(yy,V)+np.dot(z,U)+epsilon
    
    t=np.repeat(np.arange(0,M),N_i)
    if s==0:
        x_train=x
        y_train=y
        z_train=z
        t_train=t
    elif s==1:
        x_val=x
        y_val=y
        z_val=z
        t_val=t
    else:
        x_test=x
        y_test=y
        z_test=z
        t_test=t


f=h5py.File('data.h5','w')
f.create_dataset('V',data=V)
f.create_dataset('U',data=U)
f.create_dataset('mu',data=mu)
f.create_dataset('x_var',data=x_var)
f.create_dataset('x_train',data=x_train)
f.create_dataset('x_val',data=x_val)
f.create_dataset('x_test',data=x_test)
f.create_dataset('y_train',data=y_train)
f.create_dataset('y_val',data=y_val)
f.create_dataset('y_test',data=y_test)
f.create_dataset('z_train',data=z_train)
f.create_dataset('z_val',data=z_val)
f.create_dataset('z_test',data=z_test)
f.create_dataset('t_train',data=t_train)
f.create_dataset('t_val',data=t_val)
f.create_dataset('t_test',data=t_test)


plt.figure(figsize=(6, 6))
plt.scatter(x_train[:, 0], x_train[:, 1], c=t_train)
plt.colorbar()
plt.show()
plt.savefig('x_train_t.pdf')


plt.figure(figsize=(6, 6))
idx=t_train==0
plt.scatter(x_train[idx, 0], x_train[idx, 1], c='b')
plt.hold(True)
idx=t_train==1
plt.scatter(x_train[idx, 0], x_train[idx, 1], c='g')
idx=t_train==4
plt.scatter(x_train[idx, 0], x_train[idx, 1], c='r')
plt.show()
plt.savefig('x_train_t3.pdf')

print(np.mean(x_train,axis=0))
print(np.std(x_train,axis=0))
