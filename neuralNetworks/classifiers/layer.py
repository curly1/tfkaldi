'''@file layer.py
Neural network layers '''

import tensorflow as tf
from neuralNetworks import ops 
import seq_convertors
import sys

class FFLayer(object):
    '''This class defines a fully connected feed forward layer'''

    def __init__(self, output_dim, activation, weights_std=None):
        '''
        FFLayer constructor, defines the variables
        Args:
            output_dim: output dimension of the layer
            activation: the activation function
            weights_std: the standart deviation of the weights by default the
                inverse square root of the input dimension is taken
        '''

        #save the parameters
        self.output_dim = output_dim
        self.activation = activation
        self.weights_std = weights_std

    def __call__(self, inputs, is_training=False, reuse=False, scope=None):
        '''
        Do the forward computation
        Args:
            inputs: the input to the layer
            is_training: whether or not the network is in training mode
            reuse: wheter or not the variables in the network should be reused
            scope: the variable scope of the layer
        Returns:
            The output of the layer
        '''
         
        with tf.variable_scope(scope or type(self).__name__, reuse=reuse):
            with tf.variable_scope('parameters', reuse=reuse):

                stddev = (self.weights_std if self.weights_std is not None
                          else 1/int(inputs.get_shape()[1])**0.5)
            
                weights = tf.get_variable(
                    'weights', [inputs.get_shape()[1], self.output_dim],
                    initializer=tf.random_normal_initializer(stddev=stddev))

                biases = tf.get_variable(
                    'biases', [self.output_dim],
                    initializer=tf.constant_initializer(0))
            
            print "inputs", inputs
            print "weights", weights
            print "biases", biases

            #apply weights and biases
            with tf.variable_scope('linear', reuse=reuse):
                linear = tf.matmul(inputs, weights) + biases

            #apply activation function
            with tf.variable_scope('activation', reuse=reuse):
                outputs = self.activation(linear, is_training, reuse)

        return outputs

class Conv2dLayer(object):
    '''a 1-D convolutional layer'''

    def __init__(self, num_units, kernel_size, stride):
        '''constructor
        Args:
            num_units: the number of filters
            kernel_size: the size of the filters
            stride: the stride of the convolution
        '''

        self.num_units = num_units
        self.kernel_size = kernel_size
        self.stride = stride

    def __call__(self, inputs, seq_length, is_training=False, scope=None):
        '''
        Create the variables and do the forward computation
        Args:
            inputs: the input to the layer as a
                [batch_size, max_length, dim] tensor
            seq_length: the length of the input sequences
            is_training: whether or not the network is in training mode
            scope: The variable scope sets the namespace under which
                      the variables created during this call will be stored.
        Returns:
            the outputs which is a [batch_size, max_length/stride, num_units]
        '''

        with tf.variable_scope(scope or type(self).__name__):
           
            numchannels_in = int(inputs.get_shape()[3])
            input_dim = self.kernel_size * self.kernel_size * numchannels_in
            
            #print "input_dim", input_dim
            #print "num_channels_in", numchannels_in
            #print inputs.get_shape()
 
            stddev = 1/input_dim**0.5

            #the filter parameters
            w = tf.get_variable(
                'filter', [self.kernel_size, self.kernel_size, numchannels_in, self.num_units],
                initializer=tf.random_normal_initializer(stddev=stddev))

            #the bias parameters
            b = tf.get_variable(
                'bias', [self.num_units],
                initializer=tf.random_normal_initializer(stddev=stddev))
            
            #do the convolution
            out = tf.nn.conv2d(inputs, w, [1, self.stride, self.stride, 1], padding='SAME')
 
            #add the bias
            out = tf.nn.bias_add(out, b)
            
        return out


