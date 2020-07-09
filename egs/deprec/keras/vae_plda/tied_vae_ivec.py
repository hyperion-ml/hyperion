'''
VAE for  GMM i-vector distribution
Reference: "Auto-Encoding Variational Bayes" https://arxiv.org/abs/1312.6114
'''
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from keras.layers import Input, Dense, Lambda, Merge
from keras.layers.pooling import GlobalAveragePooling1D
from keras.layers.wrappers import TimeDistributed
from keras.models import Model
from keras import backend as K
from keras import objectives
from keras.datasets import mnist
from keras import initializations
from keras import optimizers
from keras.regularizers import l2

import h5py
from six.moves import xrange

from cronus.keras.vae.tied_vae_qyqz import TiedVAE_qYqZ as TVAE


def plot_cov_ellipse(cov, pos, volume=.5, ax=None, fc='none', ec=[0,0,0], a=1, lw=2):
        """
        Plots an ellipse enclosing *volume* based on the specified covariance
        matrix (*cov*) and location (*pos*). Additional keyword arguments are passed on to the 
        ellipse patch artist.
        
        Parameters
        ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
        sequence of [x0, y0].
        volume : The volume inside the ellipse; defaults to 0.5
        ax : The axis that the ellipse will be plotted on. Defaults to the 
        current axis.
        """
        from scipy.stats import chi2
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse
        
        def eigsorted(cov):
            vals, vecs = np.linalg.eigh(cov)
            order = vals.argsort()[::-1]
            return vals[order], vecs[:,order]
    
        if ax is None:
            ax = plt.gca()
        
        vals, vecs = eigsorted(cov)
        theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
        
        kwrg = {'facecolor':fc, 'edgecolor':ec, 'alpha':a, 'linewidth':lw}
        
        # Width and height are "full" widths, not radius
        width, height = 2 * np.sqrt(chi2.ppf(volume,2)) * np.sqrt(vals)
        ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwrg)

        ax.add_artist(ellip)
            
D=100
batch_size = 25
intermediate_dim = 200
epochs = 1000
l2_reg=0.0001
num_samples=1000
N_i=8

# load data
f=h5py.File('data.h5','r')

mu=np.asarray(f['mu'],dtype='float32').ravel()
V=np.asarray(f['V'],dtype='float32')
U=np.asarray(f['U'],dtype='float32')
VU=np.vstack((V,U))
x_var=np.asarray(f['x_var'],dtype='float32').ravel()
x_logvar=np.log(x_var)
x_train=np.asarray(f['x_train'],dtype='float32')
x_val=np.asarray(f['x_val'],dtype='float32')
y_train=np.asarray(f['y_train'],dtype='float32')
y_val=np.asarray(f['y_val'],dtype='float32')
t_train=np.asarray(f['t_train'],dtype='float32')
t_val=np.asarray(f['t_val'],dtype='float32')

original_dim=mu.shape[0]
y_dim=V.shape[0]
z_dim=U.shape[0]

# normalize data
# mean_x=np.mean(x_train,axis=0)
# std_x=np.std(x_train,axis=0)
# x_train=(x_train-mean_x)/std_x
# x_val=(x_val-mean_x)/std_x

# 2D to 3D tensor
x_train=np.reshape(x_train,(-1,N_i,D))
x_val=np.reshape(x_val,(-1,N_i,D))

M_val=int(np.max(t_val))+1


def my_init(shape,name=None):
    return initializations.normal(shape, scale=0.1, name=name)

# define encoder architecture
x = Input(shape=(N_i,original_dim,))
h1 = TimeDistributed(Dense(intermediate_dim, activation='relu',
                           init=my_init, W_regularizer=l2(l2_reg)))(x)
h2 = TimeDistributed(Dense(int(intermediate_dim/2), activation='relu',
                           init=my_init, W_regularizer=l2(l2_reg)))(h1)
z_mean = TimeDistributed(Dense(z_dim, init=my_init, W_regularizer=l2(l2_reg)))(h2)
z_logvar = TimeDistributed(Dense(z_dim, init=my_init, W_regularizer=l2(l2_reg)))(h2)

h3 = TimeDistributed(Dense(int(intermediate_dim/2), activation='relu',
                           init=my_init, W_regularizer=l2(l2_reg)))(h1)

h3pool = GlobalAveragePooling1D()(h3)
y_mean = Dense(y_dim, activation='relu',
               init=my_init, W_regularizer=l2(l2_reg))(h3pool)
y_logvar = Dense(y_dim, activation='relu',
                 init=my_init, W_regularizer=l2(l2_reg))(h3pool)

encoder=Model(x,[y_mean, y_logvar, z_mean, z_logvar])

# define decoder architecture
y=Input(shape=(N_i, y_dim,))
z=Input(shape=(N_i, z_dim,))
yz = Merge(mode='concat', concat_axis=-1)([y, z])
x_dec_mean = TimeDistributed(Dense(original_dim, weights=[VU, mu], trainable=False))(yz)
x_dec_logvar = TimeDistributed(Dense(original_dim, weights=[np.zeros((y_dim,D), dtype='float32'), x_logvar], trainable=False))(y)

decoder=Model([y, z],[x_dec_mean, x_dec_logvar])

# train VAE
vae=TVAE(encoder,decoder,'normal')
vae.build()
opt = optimizers.Adam(lr=0.001)
vae.train(x_train,x_val=x_val,optimizer=opt,
          shuffle=True,
          epochs=epochs,
          batch_size=batch_size)


#Colormap stuff
cm_norm_t = matplotlib.colors.Normalize(0,M_val)

# choose a colormap
c_m = matplotlib.cm.jet

# create a ScalarMappable and initialize a data structure
s_m_t = matplotlib.cm.ScalarMappable(cmap=c_m, norm=cm_norm_t)
s_m_t.set_array([])

# plot the latent space
yz = vae.compute_qyz_x(x_val, batch_size=batch_size)
y_mean_val=yz[0]
y_logvar_val=yz[1]
z_mean_val=yz[2]
z_logvar_val=yz[3]

y_mean_val2=np.reshape(y_mean_val,(-1,y_dim))
y_var_val2=np.exp(np.reshape(y_logvar_val,(-1,y_dim)))

# load gt
f=h5py.File('plda_gt.h5','r')
muy_gt_val=np.asarray(f['muy_val'],dtype='float32')
Cy_gt_val=np.asarray(f['Cy_val'],dtype='float32')

plt.figure(figsize=(6, 6))
plt.hold(True)

#y_val2=np.reshape(y_val,(-1,2))

# delta=0.025
# y1 = np.arange(-3.0, 3.0, delta)
# y2 = np.arange(-3.0, 3.0, delta)
# Y1, Y2 = np.meshgrid(y1, y2)
cc=['b', 'r', 'g', 'm', 'c', 'k']
for i in xrange(6):
    cov=np.diag(y_var_val2[i,:])
    plt.plot(y_mean_val2[i,0],y_mean_val2[i,1],cc[i]+'x')
    plot_cov_ellipse(cov=cov,pos=y_mean_val2[i,:],ec=cc[i])
    cov_gt=np.reshape(Cy_gt_val[i,:],(2,2))
    plt.plot(muy_gt_val[i,0],muy_gt_val[i,1],cc[i]+'+')
    plot_cov_ellipse(cov=cov_gt,pos=muy_gt_val[i,:],ec=cc[i],lw=4)
 #   plt.plot(y_val2[i,0],y_val2[i,1],cc[i]+'s')
    # Z=matplotlib.mlab.bivariate_normal(Y1,Y2,sigmax=y_std_val2[i,0], sigmay=y_std_val2[i,1], mux=y_mean_val2[i,0], muy=y_std_val2[i,1], sigmaxy=0)
    # CS = plt.contour(Y1, Y2, Z)
    # plt.clabel(CS, inline=1, fontsize=10)
plt.xlim(-4,4)
plt.ylim(-4,4)
plt.title('Q(y)')
plt.show()
plt.savefig('tied_vae_qy_val.pdf')
print(y_mean_val2[:6,:])
print(muy_gt_val[:6,:])
print(y_var_val2[:6,:])
print(Cy_gt_val[:6,:])

