from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

from monitor import Monitor


class TrainMonitor(Monitor):

    def __init__(self, file_path, monitor='val_loss', stop_patience=20, 
                 lr_scaling_factor=0.9, lr_patience=5, verbose=1, mode='auto'):
        super(TrainMonitor, self).__init__(file_path,verbose)
        self.monitor = monitor
        self.patience  =  stop_patience
        self.lr_scaling_factor = lr_scaling_factor
        self.lr_patience = lr_patience

        self.wait = 0
        self.lr_wait = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('EarlyStopping mode %s is unknown, '
                          'fallback to auto mode.' % (self.mode), RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

        
    def on_epoch_end(self, epoch, logs={}):
        epoch+=self.init_epoch
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn('Monitoring requires %s available!' %
                          (self.monitor), RuntimeWarning)

        assert hasattr(self.model.optimizer, 'lr'), \
            'Optimizer must have a "lr" attribute.'
        improve=self.monitor_op(current, self.best)
        lr=K.get_value(self.model.optimizer.lr)
        info_str=self.get_info_str(epoch,logs,improve,lr)
        if self.verbose>0:
            print(info_str)
            sys.stdout.flush()

        if improve:
            self.best = current
            self.wait = 0
            self.lr_wait = 0
            self.save_best(info_str)
        else:
            if self.wait >= self.patience:
                if self.verbose > 0:
                    print('epoch %05d: early stopping' % (epoch))
                self.model.stop_training = True
            self.wait += 1

            if self.lr_wait >= self.lr_patience:
                new_lr=self.lr_scaling_factor*lr
                K.set_value(self.model.optimizer.lr, new_lr)
                self.lr_wait=0
                if self.verbose > 0:
                    print('epoch %05d: reducing learning rate %f -> %f' % (epoch,lr,new_lr))
            else:
                self.lr_wait+=1
        self.save_last(epoch,info_str)

    def get_info_str(self,epoch,logs,improve,lr):
        n_epochs_no_best=0 if improve else self.wait+1
        loss=logs.get('loss')
        val_loss=logs.get('val_loss')
        acc=-1.0
        if 'acc' in logs:
            acc=logs.get('acc')
            val_acc=logs.get('val_acc')
        elif 'masked_binary_accuracy' in logs:
            acc=logs.get('masked_binary_accuracy')
            val_acc=logs.get('val_masked_binary_accuracy')

        if acc>-1.0:
            info=('epoch: %05d loss: %f '
                  'val_loss: %f '
                  'acc: %f val_acc: %f '
                  'improve: %d n_epochs_no_best: %d ' 
                  'learning_rate: %f ' 
                  'elapsed_time: %.2f secs.') % (
                      epoch,loss,val_loss,
                      acc,val_acc,
                      improve,n_epochs_no_best,lr,
                      time.time()-self.t_start)
        else:
            info=('epoch: %05d loss: %f '
                  'val_loss: %f '
                  'improve: %d n_epochs_no_best: %d ' 
                  'learning_rate: %f ' 
                  'elapsed_time: %.2f secs.') % (
                      epoch,loss,val_loss,
                      improve,n_epochs_no_best,lr,
                      time.time()-self.t_start)
        return info


    def save_state(self,name,epoch):
        file_state='%s/nn_state.%s.pickle' % (self.file_path,name)
        rng_state=self.rng.get_state()
        iteration=0
        if hasattr(self.model.optimizer,'iteration'):
            iteration=K.get_value(self.model.optimizer.iteration)
            print('Save %d' % iteration)

        f=open(file_state,'wb')
        pickle.dump([epoch,rng_state,self.best,self.wait,self.lr_wait,iteration],f)
        f.close()
        

    def load_state(self,name):
        file_state='%s/nn_state.%s.pickle' % (self.file_path,name)
        f=open(file_state,'rb')
        try:
            [epoch,rng_state,best,wait,lr_wait,iteration]=pickle.load(f,encoding='latin1')
        except TypeError:
            [epoch,rng_state,best,wait,lr_wait,iteration]=pickle.load(f)
        f.close()

        if wait >= self.patience:
            sys.exit()

        self.rng.set_state(rng_state)
        self.best=best
        self.wait=wait
        self.lr_wait=lr_wait
        self.init_epoch=epoch+1
        self.iteration=iteration
        return 

