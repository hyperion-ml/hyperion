from __future__ import absolute_import
from __future__ import print_function


from keras.optimizers import SGD, RMSprop, Adam, Adamax , Nadam

class OptimizerFactory(object):


    @staticmethod
    def create_optimizer(opt_type, lr, momentum=0, lr_decay=0.,
                         rho=0.9, epsilon=None, beta_1=0.9, beta_2=0.999, 
                         clipnorm=10, clipvalue=100, amsgrad=False):

        if opt_type == 'sgd':
            return SGD(lr=lr, momentum=momentum, decay=lr_decay, nesterov=False,
                   clipnorm=clipnorm, clipvalue=clipvalue)
        if opt_type == 'nsgd':
            return SGD(lr=lr, momentum=momentum, decay=lr_decay, nesterov=True,
                       clipnorm=clipnorm, clipvalue=clipvalue)
        if opt_type == 'rmsprop':
            return RMSprop(lr=lr, rho=rho, epsilon=epsilon, decay=lr_decay,
                           clipnorm=clipnorm, clipvalue=clipvalue)
        if opt_type == 'adam':
            return Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon,
                        decay=lr_decay, amsgrad=amsgrad,
                        clipnorm=clipnorm, clipvalue=clipvalue)
        if opt_type == 'nadam':
            return Nadam(lr=lr, beta_1=beta_1, beta_2=beta_2, 
                         epsilon=epsilon, schedule_decay=lr_decay,
                         clipnorm=clipnorm, clipvalue=clipvalue)
        if opt_type == 'adamax':
            return Adamax(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon,
                          decay=lr_decay,
                          clipnorm=clipnorm, clipvalue=clipvalue)


    @staticmethod
    def filter_args(prefix=None, **kwargs):
        if prefix is None:
            p = ''
        else:
            p = prefix + '_'
        valid_args = ('opt_type', 'lr', 'momentum', 'lr_decay',
                      'rho', 'epsilon', 'beta_1', 'beta_2',
                      'clipnorm', 'clipvalue')
        return dict((k, kwargs[p+k])
                    for k in valid_args if p+k in kwargs)
    

        
    @staticmethod
    def add_argparse_args(parser, prefix=None):
        if prefix is None:
            p1 = '--'
            p2 = ''
        else:
            p1 = '--' + prefix + '-'
            p2 = prefix + '_'

        parser.add_argument(p1+'optimizer', dest=(p2+'opt_type'), type=str.lower,
                        default='adam',
                        choices=['sgd','nsgd','rmsprop','adam','nadam','adamax'],
                        help=('Optimizers: SGD, '
                              'NSGD (SGD with Nesterov momentum), '
                              'RMSprop, Adam, Adamax, '
                              'Nadam (Adam with Nesterov momentum), '
                              '(default: %(default)s)'))

        parser.add_argument(p1+'lr' , dest=(p2+'lr'),
                            default=0.002, type=float,
                            help=('Initial learning rate (default: %(default)s)'))
        parser.add_argument(p1+'momentum', dest=(p2+'momentum'), default=0.6, type=float,
                            help=('Momentum (default: %(default)s)'))
        parser.add_argument(p1+'lr-decay', dest=(p2+'lr_decay'), default=1e-6, type=float,
                            help=('Learning rate decay in SGD optimizer '
                                  '(default: %(default)s)'))
        parser.add_argument(p1+'rho', dest=(p2+'rho'), default=0.9, type=float,
                            help=('Rho in RMSprop optimizer (default: %(default)s)'))
        parser.add_argument(p1+'epsilon', dest=(p2+'epsilon'), default=1e-8, type=float,
                            help=('Epsilon in RMSprop and Adam optimizers '
                                  '(default: %(default)s)'))
        parser.add_argument(p1+'amsgrad', dest=(p2+'amsgrad'), default=False,
                            action='store_true',
                            help=('AMSGrad variant of Adam'))

        parser.add_argument(p1+'beta1', dest=(p2+'beta_1'), default=0.9, type=float,
                            help=('Beta_1 in Adam optimizers (default: %(default)s)'))
        parser.add_argument(p1+'beta2', dest=(p2+'beta_2'), default=0.999, type=float,
                            help=('Beta_2 in Adam optimizers (default: %(default)s)'))
        # parser.add_argument(p1+'schedule-decay', dest=(p2+'schedule_decay'),
        #                     default=0.004, type=float,
        #                     help=('Schedule decay in Nadam optimizer '
        #                           '(default: %(default)s)'))
        parser.add_argument(p1+'clipnorm', dest=(p2+'clipnorm'),
                            default=10,type=float,
                            help=('Clips the norm of the gradient '
                                  '(default: %(default)s)'))
        parser.add_argument(p1+'clipvalue', dest=(p2+'clipvalue'),
                            default=100,type=float,
                            help=('Clips the absolute value of the gradient '
                            '(default: %(default)s)'))


            



    
