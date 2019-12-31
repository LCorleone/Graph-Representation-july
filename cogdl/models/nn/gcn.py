# -*- coding: utf-8 -*-
# @Author: lxy
# @Date:   2019-12-24 15:38:40
# @Last Modified by:   lxy
# @Last Modified time: 2019-12-31 10:51:19


import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
import numpy as np


class GraphConvolution(keras.layers.Layer):
    """Basic graph convolution layer as in https://arxiv.org/abs/1609.02907"""

    def __init__(self, units, support=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(GraphConvolution, self).__init__(**kwargs)
        self.units = units
        self.activation = keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.activity_regularizer = keras.regularizers.get(activity_regularizer)
        self.kernel_constraint = keras.constraints.get(kernel_constraint)
        self.bias_constraint = keras.constraints.get(bias_constraint)
        self.supports_masking = True

        self.support = support
        assert support >= 1

    def compute_output_shape(self, input_shapes):
        features_shape = input_shapes[0]
        output_shape = (features_shape[0], self.units)
        return output_shape  # (batch_size, output_dim)

    def build(self, input_shapes):
        features_shape = input_shapes[0]
        # assert len(features_shape) == 2
        input_dim = features_shape[-1]

        self.kernel = self.add_weight(shape=(input_dim * self.support,
                                             self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs, mask=None):
        features = inputs[0]
        basis = inputs[1:]

        supports = list()
        for i in range(self.support):
            supports.append(K.dot(basis[i], features))
        supports = K.concatenate(supports, axis=1)
        output = K.dot(supports, self.kernel)

        if self.use_bias:
            output += self.bias
        return self.activation(output)

    def get_config(self):
        config = {'units': self.units,
                  'support': self.support,
                  'activation': keras.activations.serialize(self.activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer': keras.initializers.serialize(
                      self.kernel_initializer),
                  'bias_initializer': keras.initializers.serialize(
                      self.bias_initializer),
                  'kernel_regularizer': keras.regularizers.serialize(
                      self.kernel_regularizer),
                  'bias_regularizer': keras.regularizers.serialize(
                      self.bias_regularizer),
                  'activity_regularizer': keras.regularizers.serialize(
                      self.activity_regularizer),
                  'kernel_constraint': keras.constraints.serialize(
                      self.kernel_constraint),
                  'bias_constraint': keras.constraints.serialize(self.bias_constraint)
                  }

        base_config = super(GraphConvolution, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class GCN(object):

    def __init__(self, args):
        super(GCN, self).__init__()
        self.hidden_size = args.hidden_size
        self.label_size = args.label_size
        self.gc1 = GraphConvolution(self.hidden_size, activation='relu')
        self.gc2 = GraphConvolution(self.label_size, activation='softmax')
        self.dropout = keras.layers.Dropout(0.5)

    def build(self, X, Adj):
        X_in = keras.Input(X.shape[1], )
        Adj_in = keras.Input(Adj.shape[0], Adj.shape[0])
        H = self.gc1([X_in, Adj_in])
        H = self.dropout(H)
        out = self.gc2([H, Adj_in])
        model = keras.Model([X_in, Adj_in], out)
        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=0.01))
        model.summary()
        return model
