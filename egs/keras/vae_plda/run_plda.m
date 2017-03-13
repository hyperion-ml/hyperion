addpath(genpath('~/usr/matlab'))

h5_file='data.h5'
mu=hdf5read(h5_file,'/mu'); 
V=hdf5read(h5_file,'/V');
U=hdf5read(h5_file,'/U');

x_var=hdf5read(h5_file,'/x_var');


x_train=hdf5read(h5_file,'/x_train');
x_val=hdf5read(h5_file,'/x_val');
y_train=hdf5read(h5_file,'/y_train');
y_val=hdf5read(h5_file,'/y_train');
t_train=hdf5read(h5_file,'/t_train');
t_val=hdf5read(h5_file,'/t_train');

% mean_x=mean(x_train,2);
% std_x=std(x_train,[],2);
% x_train=bsxfun(@rdivide,bsxfun(@minus,x_train,mean_x),std_x);
% x_val=bsxfun(@rdivide,bsxfun(@minus,x_val,mean_x),std_x);

% mu=mu-mean_x;
% V=bsxfun(@rdivide,V,std_x);
% U=bsxfun(@rdivide,U,std_x);
% x_var=x_var./(std_x.^2)
% 

D=1./x_var;

whos
model.mu=mu;
model.V=V;
model.U=U;
model.D=D;
model=PLDA(model);

[junk junk spk_ids]=unique(t_train);
stats=PLDA.compute_stats(x_train,spk_ids);
[y Cy]=model.compute_py(stats);

h5_file='plda_gt.h5'
hdf5write(h5_file,'/muy_train',y)
hdf5write(h5_file,'/Cy_train',Cy,'WriteMode','append')

[junk junk spk_ids]=unique(t_val);
stats=PLDA.compute_stats(x_val,spk_ids);
[y Cy]=model.compute_py(stats);

hdf5write(h5_file,'/muy_val',y,'WriteMode','append')
hdf5write(h5_file,'/Cy_val',Cy,'WriteMode','append')
