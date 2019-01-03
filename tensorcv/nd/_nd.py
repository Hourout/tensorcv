import tensorflow as tf

__all__ = ['softshrink', 'tanhshrink', 'softmin', 'log_softmin']

def softshrink(x, delta=0.5, name=None):
    """Applies the soft shrinkage function elementwise:
    
    if x > delta, output = x - delta;
    if x < -delta, output = x + delta;
    otherwise, output = 0
    Args:
        x: A Tensor with type float32 or float64.
        delta: the value for the Softshrink formulation. Default: 0.5
        name: A name for the operation (optional).
    Returns:
        A Tensor with the same type as x.
    """
    def f1(): return x+delta
    def f2(): return x-delta
    def f3(): return 0.
    return tf.case({tf.less(x, -delta): f1, 
                    tf.greater(x, delta): f2},
                   default=f3, exclusive=True)

def tanhshrink(x, name=None):
    """Applies the tanh shrinkage function elementwise:
    
    output = x - tanh(x)
    Args:
        x: A Tensor with type float32 or float64.
        name: A name for the operation (optional).
    Returns:
        A Tensor with the same type as x.
    """
    return tf.math.subtract(x, tf.math.tanh(x))

def softmin(x, axis=None, name=None):
    """Applies the Softmin function to an n-dimensional 
    input Tensor rescaling them so that the elements of 
    the n-dimensional output Tensor lie in the range (0, 1) and sum to 1
    
    output = exp(-x)/sum(exp(-x))
    Args:
        x: A Tensor with type float32 or float64.
        axis: The dimension softmax would be performed on. The default is -1 which indicates the last dimension.
        name: A name for the operation (optional).
    Returns:
        A Tensor with the same type as x.
    """
    return tf.exp(-x) / tf.reduce_sum(tf.exp(-x), axis)

def log_softmin(x, axis=None, name=None):
    """Applies the log Softmin function to an n-dimensional 
    input Tensor rescaling them so that the elements of 
    the n-dimensional output Tensor lie in the range (0, 1) and sum to 1
    
    output = log(exp(-x)/sum(exp(-x)))
    Args:
        x: A Tensor with type float32 or float64.
        axis: The dimension softmax would be performed on. The default is -1 which indicates the last dimension.
        name: A name for the operation (optional).
    Returns:
        A Tensor with the same type as x.
    """
    return tf.math.log(tf.exp(-x) / tf.reduce_sum(tf.exp(-x), axis))
