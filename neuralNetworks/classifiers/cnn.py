'''@file cnn.py
The CNN neural network classifier'''

import seq_convertors
import tensorflow as tf
from classifier import Classifier
from layer import FFLayer, Conv2dLayer
from activation import TfActivation
import sys

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
            #nonseq inputs for cnn
            #nonseq_inputs = seq_convertors.seq2nonseq(inputs, seq_length) #(?,440)  
            
            #print 'inputs', inputs #tensorlist 2037x(16,440)
            stacked_inputs = tf.pack(inputs, axis=1) 
            #print 'stacked_inputs', stacked_inputs #(16,2037,440)
            #sys.exit()
            #time_steps = [stacked_inputs]
            #num_time_steps = 5
           
            #for i in range(num_time_steps):
            #  forward = tf.pad(stacked_inputs[:, i+1:, :], [[0,0],[0,i+1],[0,0]])
            #  backward  = tf.pad(stacked_inputs[:, :-i-1, :], [[0,0],[i+1,0],[0,0]])
            #  time_steps += [forward, backward]
                            
            #logits = tf.pack(time_steps, axis=3)

            #print 'logits', logits # (16,2037,440,11)
            #sys.exit()

            #time_context = 5
            #feat_dim = 440
            #nonseq_inputs_cnn = seq_convertors.seq2nonseq_cnn(inputs, seq_length, time_context, feat_dim) #(?, 11, 40)
            
            #apply the input layer
            logits = tf.expand_dims(stacked_inputs, 3) #wczesniej:(16, 2037, 440, 1), teraz: (?, 11, 40, 1)
            print logits            
            for l in range(1, self.num_layers):
              logits = conv(logits, seq_length, is_training, reuse, 'convlayer' + str(l))
              logits = tf.nn.relu(logits) #(?,11,40,64)
            print logits
            #sys.exit()
            #reshape to match dnn input
            #dims = tf.shape(logits)
            #dims2 = logits.get_shape().as_list()
            #logits = tf.reshape(logits,[dims[0], dims2[1] * dims2[2] * dims2[3]]) 
            #stack all the output channels for the final layer (input_dim*num_channels))
            logits = tf.reshape(logits, logits.get_shape().as_list()[0:2] + [-1]) #(16,2037,28160)
            print logits 
            #sys.exit()
            #logits = tf.pack(tf.unpack(logits), axis=1)
            
            logits = tf.unpack(logits, axis=1)
            #print logits #2037 x (16,440,1)
           
            #convert the logits to nonsequence logits for the output layer
            logits = seq_convertors.seq2nonseq(logits, seq_length)
            #print 'logits for dnn ', logits #2037 x (16,440,1)
            #sys.exit()
 
            #logits = outlayer(logits, seq_length, is_training, 'outlayer')
            logits = outlayer(logits, is_training, reuse, 'outlayer')            

            #convert the logits to sequence logits to match expected output
            seq_logits = seq_convertors.nonseq2seq(logits, seq_length, len(inputs))

            #create a saver
            saver = tf.train.Saver()
            
            control_ops = None
       
        return seq_logits, seq_length, saver, control_ops

