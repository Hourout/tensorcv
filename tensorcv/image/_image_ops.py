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

def read_image(filename, channel=0, image_format='mix'):
    image = tf.io.read_file(filename)
    if image_format=='png':
        image = tf.image.decode_png(image, channel)
    elif image_format=='bmp':
        image = tf.image.decode_bmp(image, channel)
    elif image_format=='gif':
        image = tf.image.decode_gif(image)
    elif image_format in ["jpg", "jpeg"]:
        image = tf.image.decode_jpeg(image, channel)
    elif image_format=='mix':
        image = tf.image.decode_image(image)
    else:
        raise ValueError('image_format should be one of "mix", "jpg", "jpeg", "png", "gif", "bmp".')
    return image
