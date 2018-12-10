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

def RandomFlipLeftRight(image, random=True, seed=None):
    assert isinstance(random, bool), 'random should be bool type.'
    if random:
        image = tf.image.random_flip_left_right(image, seed=seed)
    else:
        image = tf.image.flip_left_right(image)
    return image

def RandomFlipTopBottom(image, random=True, seed=None):
    assert isinstance(random, bool), 'random should be bool type.'
    if random:
        image = tf.image.random_flip_up_down(image, seed=seed)
    else:
        image = tf.image.flip_up_down(image)
    return image

def RandomTranspose(image, random=True, seed=None):
    assert isinstance(random, bool), 'random should be bool type.'
    if random:
        r = tf.random.uniform([2], 0, 1, seed=seed)
        image = tf.case([(tf.less(r[0], r[1]), lambda: tf.image.transpose_image(image))], default=lambda: image)
    else:
        image = tf.image.transpose_image(image)
    return image

def RandomRotation(image, k=[1, 2, 3, 4], random=True, seed=None):
    assert isinstance(random, bool), 'random should be bool type.'
    if random:
        assert isinstance(k, list), 'if random is True, sublist in the [1, 2, 3, 4].'
        k_value = tf.convert_to_tensor(k)
        index = tf.argmax(tf.random.uniform([tf.shape(k_value)[0]], 0, 1))
        image = tf.image.rot90(image, k_value[index])
    else:
        assert k in [1, 2, 3, 4], 'if random is False, should be int one of [1, 2, 3, 4].'
        image = tf.image.rot90(image, k)
    return image
