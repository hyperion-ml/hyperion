"""
x-vector categorical embeddings
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

import numpy as np

from keras import backend as K
from keras import optimizers
from keras import objectives
from keras.layers import Input, Concatenate, Activation
from keras.models import Model, load_model, model_from_json

from .. import objectives as hyp_obj
from ..keras_utils import *
from ..layers import *

from ...hyp_model import HypModel


class SeqQEmbed(HypModel):

    def __init__(self, embed_net,
                 num_classes,
                 score_net=None,
                 post_pdf='diag_normal',
                 pooling_input='mean+logvar',
                 pooling_output='nat+prec',
                 min_var=0.1, kl_weight=0, **kwargs):

        super(SeqQEmbed, self).__init__(**kwargs)

        self.embed_net1 = embed_net
        self.score_net = score_net
        self.post_pdf = post_pdf
        self.num_classes = num_classes
        self.pooling_input = pooling_input
        self.pooling_output = pooling_output
        self.min_var = min_var
        self.kl_weight = kl_weight
        
        self.x_dim = None
        self.model = None
        self.embed_net = None

        
        
    def build(self, max_seq_length=None):
        self.x_dim = self.embed_net1.get_input_shape_at(0)[-1]

        x = Input(shape=(max_seq_length, self.x_dim,))
        mask = CreateMask(0)(x)
        frame_embed = self.embed_net1(x)

        q_embed = GlobalDiagNormalPostStdPriorPooling1D(
            input_format=self.pooling_input,
            output_format=self.pooling_output,
            min_var=self.min_var, name='pooling')(frame_embed+[mask])

        if self.score_net is None:
            # self.score_net = CatQScoringDiagNormalHomoPostStdPrior(
            #     self.num_classes,
            #     input_format=self.pooling_output,
            #     q_class_format='mean+prec', name='scoring')
            print(self.embed_net1.get_output_shape_at(0))
            y_dim = self.embed_net1.get_output_shape_at(0)[0][-1]
            p1_input = Input(shape=(y_dim,))
            p2_input = Input(shape=(y_dim,))
            score_input = [ p1_input, p2_input]
            score_layer = CatQScoringDiagNormalHomoPostStdPrior(
                self.num_classes,
                input_format=self.pooling_output,
                q_class_format='mean+prec', name='scoring')
            self.score_net = Model(score_input, score_layer(score_input))
            
            
        score = self.score_net(q_embed)
        y = Activation('softmax', name='pclass_x')(score)

        kl_loss = KLDivNormalVsStdNormal(input_format=self.pooling_output, name='kl_loss')(q_embed)
        
        self.model = Model(x, [y, kl_loss])
        self.model.summary()

        
    def compile(self, **kwargs):

        kl_loss=(lambda y_true, y_pred: y_pred)
        self.model.compile(loss=['categorical_crossentropy', kl_loss],
                           loss_weights = [1, self.kl_weight],
                           metrics={'pclass_x': 'accuracy'},
                           weighted_metrics={'pclass_x': 'accuracy'}, **kwargs)

        
        
    def predict(self, x, **kwargs):
        return self.model.predict(x, **kwargs)


    
    def build_embed(self, layers):
        pass
        # if isinstance(layers, str):
        #     layers = [layers]

        # pooling_model = Model(self.model.input, self.model.get_layer('pooling').output)
        # pooling = pooling_model(self.model.input)
        # outputs = []
        # x = self.model.input
        # for layer_name in layers:
        #     embed_i = Model(self.embed_net2.get_input_at(0), self.embed_net2.get_layer(layer_name).get_output_at(0))(pooling)
        #     outputs.append(embed_i)

        # self.embed_net = Model(x, outputs)


        
    @property
    def embed_dim(self):
        if self.embed_net is None:
            return None
        embed_dim=0
        for node in xrange(len(self.embed_net.inbound_nodes)):
            output_shape = self.embed_net.get_output_shape_at(node)
            if isinstance(output_shape, list):
                for shape in output_shape:
                    embed_dim += shape[-1]
            else:
                embed_dim += output_shape[-1]

        return embed_dim


    
    def predict_embed(self, x, **kwargs):
        embeds = self.embed_net.predict(x, **kwargs)
        return np.hstack(tuple(embeds))

    
    
    def fit(**kwargs):
        self.model.fit(**kwargs)


        
    def fit_generator(self, generator, steps_per_epoch, **kwargs):
        self.model.fit_generator(generator, steps_per_epoch, **kwargs)


        
    def get_config(self):
        config = { 'num_classes': self.num_classes,
                   'post_pdf': self.post_pdf,
                   'pooling_input' : self.pooling_input,
                   'pooling_output': self.pooling_output,
                   'min_var': self.min_var }
        base_config = super(SeqQEmbed, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


    
    def save(self, file_path):
        file_model = '%s.json' % (file_path)
        with open(file_model, 'w') as f:
            f.write(self.to_json())
        
        file_model = '%s.net1.h5' % (file_path)
        self.embed_net1.save(file_model)
        file_model = '%s.score.h5' % (file_path)
        # if not isinstance(self.score_net, Model):
        #     x
        #     self.score_net=Model(self.model.get_layer('scoring').get_input_at(0), self.model.get_layer('scoring').get_output_at(0))
        self.score_net.save(file_model)



        

    @classmethod
    def load(cls, file_path):
        file_config = '%s.json' % (file_path)
        with open(file_config,'r') as f:
            config=SeqQEmbed.load_config_from_json(f.read())

        file_model = '%s.net1.h5' % (file_path)
        embed_net1 = load_model(file_model, custom_objects=get_keras_custom_obj())
        file_model = '%s.score.h5' % (file_path)
        score_net = load_model(file_model, custom_objects=get_keras_custom_obj())

        filter_args = ('post_pdf', 'pooling_input', 'pooling_output', 'min_var', 'name')
        kwargs = {k: config[k] for k in filter_args if k in config }
        return cls(embed_net1, config['num_classes'], score_net, **kwargs)

    
    
    
