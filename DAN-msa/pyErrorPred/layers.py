import tensorflow as tf

# Peforms pixel wise spatial self-attention on an input matrix.
def pixelSelfAttention(x,
                       maxpool=1,
                       kq_factor=8,
                       v_factor=2,
                       reuse=False):
    
    # Get parameters
    bs, _, _, c = x.get_shape().as_list()
    height = tf.shape(x)[1]
    width = tf.shape(x)[2]
    
    # Key
    f = tf.layers.conv2d(inputs=x,
                         filters=c//kq_factor,
                         kernel_size=1,
                         strides=1)
    
    if maxpool > 1: 
        f = tf.layers.max_pooling2d(f,
                                    pool_size=maxpool,
                                    strides=maxpool,
                                    padding='SAME')
    # Query
    g = tf.layers.conv2d(inputs=x,
                         filters=c//kq_factor,
                         kernel_size=1,
                         strides=1)
    
    # Value
    h = tf.layers.conv2d(inputs=x,
                         filters=c//v_factor,
                         kernel_size=1,
                         strides=1)
    
    if maxpool > 1: 
        h = tf.layers.max_pooling2d(h,
                                    pool_size=maxpool,
                                    strides=maxpool,
                                    padding='SAME')
    print(h)
        
    def hw_flatten(matrix) :
        return tf.reshape(matrix, shape=[matrix.shape[0], -1, matrix.shape[-1]])
        
    # Flattening and generating attention_map
    # Attention map: this should suppossed to be N by N whre N=hxc 
    attent_map = tf.matmul(hw_flatten(g),
                           hw_flatten(f),
                           transpose_b=True)
    beta = tf.nn.softmax(attent_map)
    print(beta)

    # Calculated infuluence 
    o = tf.matmul(beta,
                  hw_flatten(h))
    o = tf.reshape(o, shape=[bs, height, width, c//v_factor])
    o = tf.layers.conv2d(inputs=o,
                         filters=c,
                         kernel_size=1,
                         strides=1)
    
    # Initialize gamma to be 0.
    gamma = tf.Variable(0, dtype=tf.float32)
    
    # Adding back using gamma
    x = gamma * o + x
    return x