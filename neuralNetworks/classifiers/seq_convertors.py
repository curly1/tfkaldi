'''@file seq_convertors.py
this file contains functions that convert sequential data to non-sequential data
and the other way around. Sequential data is defined to be data that is suetable
as RNN input. This means that the data is a list containing an N x F tensor for
each time step where N is the batch size and F is the input dimension non
sequential data is data suetable for input to fully connected layers. This means
that the data is a TxF tensor where T is the sum of all sequence lengths. This
functionality only works for q specified batch size'''

import tensorflow as tf
import sys

def seq2nonseq(tensorlist, seq_length, name=None):
    '''
    Convert sequential data to non sequential data

    Args:
        tensorlist: the sequential data, wich is a list containing an N x F
            tensor for each time step where N is the batch size and F is the
            input dimension
        seq_length: a vector containing the sequence lengths
        name: [optional] the name of the operation

    Returns:
        non sequential data, which is a TxF tensor where T is the sum of all
        sequence lengths
    '''

    with tf.name_scope(name or 'seq2nonseq'):
        #convert the list for each time step to a list for each sequence
        sequences = tf.unpack(tf.pack(tensorlist), axis=1)

        #remove the padding from sequences
        sequences = [tf.gather(sequences[s], tf.range(seq_length[s]))
                     for s in range(len(sequences))]
        #concatenate the sequences
        tensor = tf.concat(0, sequences)

    return tensor

def seq2nonseq_cnn(tensorlist, seq_length, time_context, feat_dim, name=None):
    '''
    Convert sequential data to non sequential data

    Args:
        tensorlist: the sequential data, wich is a list containing an N x F
            tensor for each time step where N is the batch size and F is the
            input dimension
        seq_length: a vector containing the sequence lengths
        name: [optional] the name of the operation

    Returns:
        non sequential data, which is a TxF tensor where T is the sum of all
        sequence lengths
    '''

    with tf.name_scope(name or 'seq2nonseq_cnn'):
        #convert the list for each time step to a list for each sequence
        sequences = tf.unpack(tf.pack(tensorlist), axis=1) #16x(2037,440)
        
        #reshape sequences for cnn 2d layer
        #packedtensorlist = tf.pack(tensorlist)
        #shape = packedtensorlist.get_shape().as_list()
        #sequences =  tf.reshape(packedtensorlist,[shape[0],shape[1],time_context*2+1,feat_dim/(time_context*2+1)]) #(2037,16,11,40)
        #sequences = tf.unpack(sequences, axis=1)
        
        #print sequences[0]
        #remove the padding from sequences
        sequences = [tf.gather(sequences[s], tf.range(seq_length[s]))
          for s in range(len(sequences))] #for dnn: 16x(?,440), for cnn: 16x(?,11,40)
        
        sequences = [tf.reshape(sequences[s], (-1,time_context*2+1,feat_dim/(time_context*2+1)))
          for s in range(len(sequences))]
        
        #concatenate the sequences
        tensor = tf.concat(0, sequences) #(?,11,40)
        #tensor = tf.expand_dims(tensor, -1) #(?,11,40,1)

    return tensor

def nonseq_cnn2nonseq_dnn(tensor, seq_length, length, name=None):

    with tf.name_scope(name or'nonseq2seq'):
       
        #get the cumulated sequence lengths to specify the positions in tensor
        cum_seq_length = tf.concat(0, [tf.constant([0]), tf.cumsum(seq_length)])
       
        #get the indices in the tensor for each sequence
        indices = [tf.range(cum_seq_length[l], cum_seq_length[l+1])
                   for l in range(int(seq_length.get_shape()[0]))]

        #create the non-padded sequences
        sequences = [tf.gather(tensor, i) for i in indices]
        print sequences
        #print tf.pad(sequences[0],[[0,length-seq_length],[0,0]])
        sys.exit()
        #pad the sequences with zeros
        sequences = [tf.pad(sequences[s], [[0, length-seq_length[s]], [0, 0]])
                     for s in range(len(sequences))]
        
        #specify that the sequences have been padded to the constant length
        for seq in sequences:
            seq.set_shape([length, int(tensor.get_shape()[1])])

        #convert the list for eqch sequence to a list for eqch time step
        tensorlist = tf.unpack(tf.pack(sequences), axis=1)

    return tensorlist
def nonseq2seq(tensor, seq_length, length, name=None):
    '''
    Convert non sequential data to sequential data

    Args:
        tensor: non sequential data, which is a TxF tensor where T is the sum of
            all sequence lengths
        seq_length: a vector containing the sequence lengths
        length: the constant length of the output sequences
        name: [optional] the name of the operation

    Returns:
        sequential data, wich is a list containing an N x F
        tensor for each time step where N is the batch size and F is the
        input dimension
    '''

    with tf.name_scope(name or'nonseq2seq'):
        #get the cumulated sequence lengths to specify the positions in tensor
        cum_seq_length = tf.concat(0, [tf.constant([0]), tf.cumsum(seq_length)])

        #get the indices in the tensor for each sequence
        indices = [tf.range(cum_seq_length[l], cum_seq_length[l+1])
                   for l in range(int(seq_length.get_shape()[0]))]

        #create the non-padded sequences
        sequences = [tf.gather(tensor, i) for i in indices]

        #pad the sequences with zeros
        sequences = [tf.pad(sequences[s], [[0, length-seq_length[s]], [0, 0]])
                     for s in range(len(sequences))]

        #specify that the sequences have been padded to the constant length
        for seq in sequences:
            seq.set_shape([length, int(tensor.get_shape()[1])])

        #convert the list for eqch sequence to a list for eqch time step
        tensorlist = tf.unpack(tf.pack(sequences), axis=1)

    return tensorlist
