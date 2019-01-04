import tensorflow as tf
from tensorflow.python.ops.image_ops_impl import _ImageDimensions

__all __ = ['euclidean', 'euclidean_standardized', 'manhattan', 'chebyshev', 'minkowski']

def euclidean(x, y, name=None):
    """
    Args:
        x: A Tensor with type float32 or float64.
        y: A Tensor with the same type as x.
        name: A name for the operation (optional).
    Returns:
        A Tensor with the same type as x.
    """
    assert _ImageDimensions(x, x.get_shape().ndims)==_ImageDimensions(y, y.get_shape().ndims), "x and y should be same shape."
    d = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(tf.math.subtract(x, y))))
    return d

def euclidean_standardized(x, y, name=None):
    """
    Args:
        x: A Tensor with type float32 or float64.
        y: A Tensor with the same type as x.
        name: A name for the operation (optional).
    Returns:
        A Tensor with the same type as x.
    """
    assert _ImageDimensions(x, x.get_shape().ndims)==_ImageDimensions(y, y.get_shape().ndims), "x and y should be same shape."
    std = tf.keras.backend.std(tf.concat([x, y], axis=-1))
    d = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(tf.math.subtract(x, y)/std)))
    return d

def manhattan(x, y, name=None):
    """
    Args:
        x: A Tensor with type float32 or float64.
        y: A Tensor with the same type as x.
        name: A name for the operation (optional).
    Returns:
        A Tensor with the same type as x.
    """
    assert _ImageDimensions(x, x.get_shape().ndims)==_ImageDimensions(y, y.get_shape().ndims), "x and y should be same shape."
    d = tf.math.reduce_sum(tf.math.abs(tf.math.subtract(x, y)))
    return d

def chebyshev(x, y, name=None):
    """
    Args:
        x: A Tensor with type float32 or float64.
        y: A Tensor with the same type as x.
        name: A name for the operation (optional).
    Returns:
        A Tensor with the same type as x.
    """
    assert _ImageDimensions(x, x.get_shape().ndims)==_ImageDimensions(y, y.get_shape().ndims), "x and y should be same shape."
    d = tf.math.reduce_max(tf.math.subtract(x, y))
    return d

def minkowski(x, y, p, name=None):
    """
    Args:
        x: A Tensor with type float32 or float64.
        y: A Tensor with the same type as x.
        name: A name for the operation (optional).
    Returns:
        A Tensor with the same type as x.
    """
    assert _ImageDimensions(x, x.get_shape().ndims)==_ImageDimensions(y, y.get_shape().ndims), "x and y should be same shape."
    d = tf.math.sqrt(tf.math.reduce_sum(tf.math.pow(tf.math.abs(tf.math.subtract(x, y)), p)))
    return d

def hamming(x, y, name=None):
    """
    Args:
        x: A Tensor with type float32 or float64.
        y: A Tensor with the same type as x.
        name: A name for the operation (optional).
    Returns:
        A Tensor with the same type as x.
    """
    assert _ImageDimensions(x, x.get_shape().ndims)==_ImageDimensions(y, y.get_shape().ndims), "x and y should be same shape."
    d = tf.math.reduce_min(tf.shape(tf.where(tf.math.equal(x, y))))
    return d
