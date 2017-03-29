'''@file cnn.py
The CNN neural network classifier'''

import seq_convertors
import tensorflow as tf
from classifier import Classifier
from layer import FFLayer, Conv2dLayer
from activation import TfActivation

class CNN(Classifier):
    '''This class is a graph for feedforward fully connected neural nets.'''

    def __init__(self, output_dim, num_layers, num_units, activation,
                 layerwise_init=True):
        '''
        CNN constructor

        Args:
            output_dim: the DNN output dimension
            num_layers: number of hidden layers
            num_units: number of hidden units
            activation: the activation function
            layerwise_init: if True the layers will be added one by one,
                otherwise all layers will be added to the network in the
                beginning
        '''

        #super constructor
        super(CNN, self).__init__(output_dim)

        #save all the DNN properties
        self.num_layers = num_layers
        self.num_units = num_units
        self.activation = activation
        self.layerwise_init = layerwise_init

    def __call__(self, inputs, seq_length, is_training=False, reuse=False,
                 scope=None):
        '''
        Add the CNN variables and operations to the graph

        '''

        with tf.variable_scope(scope or type(self).__name__, reuse=reuse):

            #input layer
            conv = Conv2dLayer(self.num_units, 3, 1)
            
            #output layer
            outlayer = FFLayer(self.output_dim,
                              TfActivation(None, lambda(x): x), 0)


            #apply the input layer
            for l in range(1, self.num_layers):
              logits = conv(inputs, seq_length, is_training, 'convlayer' + str(l))
              logits = tf.nn.relu(logits)

            logits = outlayer(logits, seq_length, is_training, 'outlayer')
            
            #convert the logits to sequence logits to match expected output
            seq_logits = seq_convertors.nonseq2seq(logits, seq_length, len(inputs))

            #create a saver
            saver = tf.train.Saver()
            
            control_ops = None

       
        return seq_logits, seq_length, saver, control_ops


class Callable(object):
    '''A class for an object that is callable'''

    def __init__(self, value):
        '''
        Callable constructor

        Args:
            tensor: a tensor
        '''

        self.value = value

    def __call__(self):
        '''
        get the object

        Returns:
            the object
        '''

        return self.value
