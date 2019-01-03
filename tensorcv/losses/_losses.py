import tensorflow as tf
K = tf.keras.backend

__all__ = ['hard_sigmoid_cross_entropy']
def hard_sigmoid_cross_entropy(labels, logits, from_logits=False):
    """Computes hard sigmoid cross entropy given `logits`.
    Args:
        labels: A `Tensor` of the same type and shape as `logits`.
        logits: A `Tensor` of type `float32` or `float64`.
        name: A name for the operation (optional).
    Returns:
        A `Tensor` of the same shape as `logits` with the componentwise
        logistic losses.
    Raises:
        ValueError: If `logits` and `labels` do not have the same shape.
    """
    if from_logits:
        output = K.hard_sigmoid(logist)
    epsilon_ = tf.convert_to_tensor(K.epsilon(), output.dtype.base_dtype)
    output = tf.clip_by_value(output, epsilon_, 1 - epsilon_)
    a = tf.math.subtract(labels, tf.math.multiply(-1, tf.math.log(output)))
    b = tf.math.multiply(tf.math.subtract(1, labels), tf.math.log(tf.math.subtract(1, output)))
    return tf.math.add(a, b)
