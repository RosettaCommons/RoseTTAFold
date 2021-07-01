import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import numpy as np
from .layers import *

# Constructs a graph of resnet block
# Default input is channles last.
def resnet_block(_input,
                 isTraining=None,
                 channel=128,
                 require_bn=False,
                 require_in=False,
                 dilation_rate=(1,1),
                 kernel_size=(3,3),
                 data_format="channels_last",
                 activation=tf.nn.elu):
    
    # Figuring out channel size
    if channel%2 != 0:
        print("Even number channels are required.")
        return -1
    down_channel = int(channel/2)
    
    # Figuring out bn axis
    if data_format=="channels_first":
        bn_axis = 1
    elif data_format=="channels_last":
        bn_axis = 3
    
    ##############################################
    # Full pre-activation style:                 #
    # bn -> elu -> conv2d -> bn -> elu -> conv2d #
    # bn -> elu -> conv2d                        #
    ##############################################
    
    if require_bn: _input = tf.layers.batch_normalization(_input,
                                                          training=isTraining,
                                                          center=True,
                                                          scale=True,
                                                          axis=bn_axis)
    if require_in: _input = tf.contrib.layers.instance_norm(_input)
        
    _input = activation(_input)
    _input = tf.layers.conv2d(_input,
                              filters=down_channel,
                              kernel_size=(1,1),
                              data_format=data_format)
    
    if require_bn: _input = tf.layers.batch_normalization(_input,
                                                          training=isTraining,
                                                          center=True,
                                                          scale=True,
                                                          axis=bn_axis)
    if require_in: _input = tf.contrib.layers.instance_norm(_input)
        
    _input = activation(_input)
    _input = tf.layers.conv2d(_input,
                              filters=down_channel, 
                              kernel_size=kernel_size, 
                              dilation_rate=dilation_rate,
                              data_format=data_format,
                              padding="same")
    
    if require_bn: _input = tf.layers.batch_normalization(_input,
                                                          training=isTraining,
                                                          center=True,
                                                          scale=True,
                                                          axis=bn_axis)
    if require_in: _input = tf.contrib.layers.instance_norm(_input)
        
    _input = activation(_input)
    _input = tf.layers.conv2d(_input,
                              filters=channel, 
                              kernel_size=(1,1),
                              data_format=data_format)
    return _input

# Creates a resnet architecture.
def build_resnet(_input,
                 channel,
                 num_chunks,
                 isTraining=None,
                 require_bn=False, #Batchnorm flag
                 require_in=False, #Instancenorm flag
                 data_format="channels_last",
                 first_projection=True,
                 no_last_dilation=False,
                 transpose_matrix=False,
                 dilation_cycle = [1,2,4,8]):
    
    # Projection of the very first input to 128 channels.
    if first_projection:
        _input = tf.layers.conv2d(_input,
                                  filters=channel,
                                  kernel_size=(1,1),
                                  dilation_rate=(1,1),
                                  data_format=data_format)
    
    # each chunk contatins 4 blocks with cycling dilation rates.
    for i in range(num_chunks):
        # dilation rates
        for dr in dilation_cycle:
            # save residual connection
            _residual = _input
            # pass through resnet block
            _conved = resnet_block(_input,
                                   isTraining=isTraining,
                                   require_bn=require_bn,
                                   require_in=require_in,
                                   channel=channel,
                                   dilation_rate=(dr, dr),
                                   data_format=data_format)
            # genearte input to the next block
            _input = _residual+_conved
            if transpose_matrix:
                _input = (_input+tf.transpose(_input, [0,2,1,3]))/2
            
    
    # 2 more extra blocks with dilation
    if no_last_dilation:
        for i in range(2):
            _residual = _input
            # pass through resnet block
            _conved = resnet_block(_input,
                                   isTraining=isTraining,
                                   require_bn=require_bn,
                                   require_in=require_in,
                                   channel=channel,
                                   dilation_rate=(1, 1),
                                   data_format=data_format)
            # genearte input to the next block
            _input = _residual+_conved
            if transpose_matrix:
                _input = (_input+tf.transpose(_input, [0,2,1,3]))/2
            
    return _input