import tensorflow as tf

def random_brightness(image, lower, upper, seed=None):
    if upper>1 or upper<=lower or lower<0:
        raise ValueError('lower and upper should be upper > lower and in the range [0,1).')
    delta = tf.random.uniform([], lower, upper, seed=seed)
    return tf.image.adjust_brightness(image, delta)

def random_hue(image, lower, upper, seed=None):
    if upper>1 or upper<=lower or lower<-1:
        raise ValueError('lower and upper should be upper > lower and in the range [-1,1].')
    delta = tf.random.uniform([], lower, upper, seed=seed)
    return tf.image.adjust_hue(image, delta)
