from __future__ import absolute_import
from __future__ import print_function

from keras import objectives
from keras.models import model_from_json
from keras.optimizers import SGD, RMSprop, Adam, Adamax , Nadam
from keras.callbacks import *
from keras.engine.training import slice_X

# from hyperion.keras.callbacks import *
# from hyperion.keras.layers.masking import *
# from hyperion.keras.layers.sampling import *
# from hyperion.keras.layers.pooling import *
# from hyperion.keras.layers.tensor_manipulation import *

from .callbacks import *
from .layers.core import *
from .layers.masking import *
from .layers.sampling import *
from .layers.cov import *
from .layers.pooling import *
from .layers.tensor_manipulation import *



def get_keras_custom_obj():
    custom_obj = {
        'Bias': Bias,
        'Constant': Constant,
        'TiledConstant': TiledConstant,
        'ConstTriu': ConstTriu,
        'TiledConstTriu': TiledConstTriu,
        'Invert': Invert,
        'Exp': Exp,
        'ExpTaylor': ExpTaylor,
        'Log1': Log1,
        'NegLog1': NegLog1,
        'Repeat': Repeat,
        'CreateMask': CreateMask,
        'GlobalMaskedAveragePooling1D':  GlobalMaskedAveragePooling1D,
        'GlobalWeightedAveragePooling1D': GlobalWeightedAveragePooling1D,
        'GlobalWeightedSumPooling1D': GlobalWeightedSumPooling1D,
        'GlobalSumPooling1D': GlobalSumPooling1D,
        'GlobalSumWeights': GlobalSumWeights,
        'GlobalProdRenormDiagNormalCommonCovStdPrior': GlobalProdRenormDiagNormalCommonCovStdPrior,
        'GlobalProdRenormDiagNormalConstCovStdPrior': GlobalProdRenormDiagNormalConstCovStdPrior,
        'GlobalProdRenormDiagNormalConstCovStdPrior2': GlobalProdRenormDiagNormalConstCovStdPrior2,
        'GlobalProdRenormDiagNormalConstCovStdPrior3': GlobalProdRenormDiagNormalConstCovStdPrior3,
        'GlobalProdRenormDiagNormalConstCovStdPrior4': GlobalProdRenormDiagNormalConstCovStdPrior4,
        'GlobalProdRenormNormalConstCovStdPrior': GlobalProdRenormNormalConstCovStdPrior,
        'MultConstDiagCov': MultConstDiagCov,
        'MultConstDiagCovStdPrior': MultConstDiagCovStdPrior,
        'MultConstCovStdPrior': MultConstCovStdPrior,
        'BernoulliSampler': BernoulliSampler,
        'DiagNormalSampler': DiagNormalSampler,
        'DiagNormalSamplerFromSeqLevel': DiagNormalSamplerFromSeqLevel,
        'Repeat': Repeat,
        'ExpandAndTile': ExpandAndTile}
    return custom_obj


def load_model_arch(file_path):
    return model_from_json(open(file_path,'r').read(), get_keras_custom_obj())


def save_model_arch(file_path, model):
    open(file_path,'w').write(model.to_json())


def filter_optimizer_args(**kwargs):
    return dict((k, kwargs[k])
                for k in ('opt_type', 'lr', 'momentum', 'decay',
                          'rho', 'epsilon', 'beta_1', 'beta_2', 
                          'clipnorm', 'clipvalue') if k in kwargs)
    
    
def create_optimizer(opt_type, lr, momentum=0, decay=0.,
                     rho=0.9, epsilon=0., beta_1=0.9, beta_2=0.999, 
                     clipnorm=10, clipvalue=100):

    if opt_type == 'sgd':
        return SGD(lr=lr, momentum=momentum, decay=decay, nesterov=False,
                   clipnorm=clipnorm, clipvalue=clipvalue)
    if opt_type == 'nsgd':
        return SGD(lr=lr, momentum=momentum, decay=decay, nesterov=True,
                   clipnorm=clipnorm, clipvalue=clipvalue)
    if opt_type == 'rmsprop':
        return RMSprop(lr=lr, rho=rho, epsilon=epsilon,
                       clipnorm=clipnorm, clipvalue=clipvalue)
    if opt_type == 'adam':
        return Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon,
                    clipnorm=clipnorm, clipvalue=clipvalue)
    if opt_type == 'nadam':
        return Nadam(lr=lr, beta_1=beta_1, beta_2=beta_2, 
                     epsilon=epsilon, schedule_decay=decay,
                     clipnorm=clipnorm, clipvalue=clipvalue)
    if opt_type == 'adamax':
        return Adamax(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon,
                      clipnorm=clipnorm, clipvalue=clipvalue)

    
def filter_callbacks_args(**kwargs):
    return dict((k, kwargs[k])
                for k in ('save_best_only', 'mode',
                          'monitor', 'patience', 'min_delta',
                          'lr_steps', 'lr_patience', 'lr_factor',
                          'min_lr', 'log_append') if k in kwargs)
    

def create_basic_callbacks(model, file_path, save_best_only=True, mode='min',
                           monitor = 'val_loss', patience=None, min_delta=1e-4,
                           lr_steps = None,
                           lr_patience = None, lr_factor=0.1, min_lr=1e-5,
                           log_append=False):

    if save_best_only == True:
        file_path_model = file_path + '/model.best'
    else:
        file_path_model = file_path + '/model.{epoch:04d}'
    cb = HypModelCheckpoint(model, file_path_model, monitor=monitor, verbose=1,
                            save_best_only=save_best_only,
                            save_weights_only=False, mode=mode)
    cbs = [cb]

    file_path_csv = file_path + '/train.log'
    cb = CSVLogger(file_path_csv, separator=',', append=log_append)
    cbs.append(cb)
    
    if patience is not None:
        cb = EarlyStopping(monitor=monitor, patience=patience,
                           min_delta=min_delta, verbose=1, mode=mode)
        cbs.append(cb)
        
    if lr_steps is not None:
        cb = LearningRateSteps(lr_steps)
        cbs.append(cb)    

    if lr_patience is not None:
        cb = ReduceLROnPlateau(monitor=monitor,
                               factor=lr_factor, patience=lr_patience,
                               verbose=1, mode=mode, epsilon=min_delta,
                               cooldown=0, min_lr=min_lr)
        cbs.append(cb)    
        
    return cbs


def weighted_objective_per_sample(fn):
    '''Transforms an objective function `fn(y_true, y_pred)`
    into a sample-weighted, cost-masked objective function
    `fn(y_true, y_pred, weights, mask)`.
    '''
    def weighted(y_true, y_pred, weights):
        # score_array has ndim >= 2
        score_array = fn(y_true, y_pred)

        # reduce score_array to same ndim as weight array
        ndim = K.ndim(score_array)
        weight_ndim = K.ndim(weights)
        score_array = K.mean(score_array, axis=list(range(weight_ndim, ndim)))

        axis=list(range(1, weight_ndim))
        # apply sample weighting
        if weights is not None:
            score_array *= weights
            score_array /= K.mean(weights, axis=axis, keepdims=True) 
        return K.mean(score_array, axis=axis)
    return weighted



def make_eval_function(model, loss, loss_weights=None, **kwargs):

    # prepare loss weights
    if loss_weights is None:
        loss_weights_list = [1. for _ in range(len(model.outputs))]
    elif isinstance(loss_weights, dict):
        for name in loss_weights:
            if name not in model.output_names:
                raise ValueError('Unknown entry in loss_weights '
                                 'dictionary: "' + name + '". '
                                 'Only expected the following keys: ' +
                                 str(model.output_names))
        loss_weights_list = []
        for name in model.output_names:
            loss_weights_list.append(loss_weights.get(name, 1.))
    elif isinstance(loss_weights, list):
        if len(loss_weights) != len(model.outputs):
            raise ValueError('When passing a list as loss_weights, '
                             'it should have one entry per model outputs. '
                             'The model has ' + str(len(model.outputs)) +
                             ' outputs, but you passed loss_weights=' +
                             str(loss_weights))
        loss_weights_list = loss_weights
    else:
        raise TypeError('Could not interpret loss_weights argument: ' +
                        str(loss_weights) +
                        ' - expected a list of dicts.')

    # prepare loss functions
    if isinstance(loss, dict):
        for name in loss:
            if name not in model.output_names:
                raise ValueError('Unknown entry in loss '
                                 'dictionary: "' + name + '". '
                                 'Only expected the following keys: ' +
                                 str(model.output_names))
        loss_functions = []
        for name in model.output_names:
            if name not in loss:
                raise ValueError('Output "' + name +
                                 '" missing from loss dictionary.')
            loss_functions.append(objectives.get(loss[name]))
    elif isinstance(loss, list):
        if len(loss) != len(model.outputs):
            raise ValueError('When passing a list as loss, '
                             'it should have one entry per model outputs. '
                             'The model has ' + str(len(model.outputs)) +
                             ' outputs, but you passed loss=' +
                             str(loss))
        loss_functions = [objectives.get(l) for l in loss]
    else:
        loss_function = objectives.get(loss)
        loss_functions = [loss_function for _ in range(len(model.outputs))]
    weighted_losses = [weighted_objective_per_sample(fn) for fn in loss_functions]

    # compute total loss
    total_loss = None
    for i in range(len(model.outputs)):
        y_true = model.targets[i]
        y_pred = model.outputs[i]
        weighted_loss = weighted_losses[i]
        sample_weight = model.sample_weights[i]
        loss_weight = loss_weights_list[i]
        output_loss = weighted_loss(y_true, y_pred, sample_weight)
        if total_loss is None:
            total_loss = loss_weight * output_loss
        else:
            total_loss += loss_weight * output_loss

    if model.uses_learning_phase and not isinstance(K.learning_phase(), int):
        inputs = model.inputs + model.targets + model.sample_weights + [K.learning_phase()]
    else:
        inputs = model.inputs + model.targets + model.sample_weights
        # return loss and metrics, no gradient updates.
        # Does update the network states.
    eval_function = K.function(inputs,
                               [total_loss],
                               updates=model.state_updates,
                               **kwargs)

    return eval_function


def make_batches(size, batch_size):
    '''Returns a list of batch indices (tuples of indices).
    '''
    nb_batch = int(np.ceil(size / float(batch_size)))
    return [(i * batch_size, min(size, (i + 1) * batch_size))
            for i in range(0, nb_batch)]



def _eval_loop(f, ins, batch_size=32):
    '''Abstract method to loop over some data in batches.
    
    # Arguments
    f: Keras function returning a list of tensors.
    ins: list of tensors to be fed to `f`.
    batch_size: integer batch size.
    verbose: verbosity mode.

    # Returns
    Scalar loss (if the model has a single output and no metrics)
    or list of scalars (if the model has multiple outputs
    and/or metrics). The attribute `model.metrics_names` will give you
    the display labels for the scalar outputs.
    '''
    nb_sample = ins[0].shape[0]
    outs = []

    batches = make_batches(nb_sample, batch_size)
    index_array = np.arange(nb_sample)
    for batch_index, (batch_start, batch_end) in enumerate(batches):
        batch_ids = index_array[batch_start:batch_end]
        if isinstance(ins[-1], float):
            # do not slice the training phase flag
            ins_batch = slice_X(ins[:-1], batch_ids) + [ins[-1]]
        else:
            ins_batch = slice_X(ins, batch_ids)

        batch_outs = f(ins_batch)
        if isinstance(batch_outs, list):
            if batch_index == 0:
                for batch_out in enumerate(batch_outs):
                    outs.append(np.zeros((nb_sample,)))
            for i, batch_out in enumerate(batch_outs):
                outs[i][batch_ids] = batch_out
        else:
            if batch_index == 0:
                outs.append(np.zeros((nb_sample,)))
            outs[i][batch_ids] = batch_out

    if len(outs) == 1:
        return outs[0]
    return outs


def eval_loss(model, loss_function, x, y, batch_size=32, sample_weight=None):

    x, y, sample_weights = model._standardize_user_data(
        x, y,
        sample_weight=sample_weight,
        check_batch_axis=False,
        batch_size=batch_size)
    # prepare inputs, delegate logic to _test_loop
    if model.uses_learning_phase and not isinstance(K.learning_phase, int):
        ins = x + y + sample_weights + [0.]
    else:
        ins = x + y + sample_weights
        
    return _eval_loop(loss_function, ins, batch_size=batch_size)

