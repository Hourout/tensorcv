import tensorflow as tf

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

def RandomBrightness(image, delta, seed=None):
    assert isinstance(delta, (int, float, list, tuple)), 'delta should be one of int, float, list, tuple.'
    if isinstance(delta, (int, float)):
        image = tf.image.adjust_brightness(image, delta)
    elif 0<=delta[0]<delta[1]:
        random_delta = tf.random.uniform([], delta[0], delta[1], seed=seed)
        image = tf.image.adjust_brightness(image, random_delta)
    else:
        raise ValueError('lower and upper should be upper > lower >= 0.')
    return image

def RandomContrast(image, delta, seed=None):
    assert isinstance(delta, (int, float, list, tuple)), 'delta should be one of int, float, list, tuple.'
    if isinstance(delta, (int, float)):
        if delta>=0:
            image = tf.image.adjust_contrast(image, delta)
        else:
            raise ValueError('if delta type one of int or float, should be delta>=0')
    elif 0<=delta[0]<delta[1]:
        image = tf.image.random_contrast(image, delta[0], delta[1], seed=seed)
    else:
        raise ValueError('if delta type one of tuple or list, lower and upper should be upper > lower >= 0.')
    return image

def RandomHue(image, delta, seed=None):
    assert isinstance(delta, (int, float, list, tuple)), 'delta should be one of int, float, list, tuple.'
    if isinstance(delta, (int, float)):
        if -1<=delta<=1:
            image = tf.image.adjust_hue(image, delta)
        else:
            raise ValueError('if delta type one of int or float, must be in the interval [-1, 1].')
    elif -1<=delta[0]<delta[1]<=1:
        image = tf.image.random_hue(image, delta[0], delta[1], seed=seed)
    else:
        raise ValueError('if delta type one of tuple or list, lower and upper should be 1 >= upper > lower >= -1.')
    return image

def RandomSaturation(image, delta, seed=None):
    assert isinstance(delta, (int, float, list, tuple)), 'delta should be one of int, float, list, tuple.'
    if isinstance(delta, (int, float)):
        if delta>=0:
            image = tf.image.adjust_saturation(image, delta)
        else:
            raise ValueError('if delta type one of int or float, should be delta>=0')
    elif 0<=delta[0]<delta[1]:
        image = tf.image.random_saturation(image, delta[0], delta[1], seed=seed)
    else:
        raise ValueError('if delta type one of tuple or list, lower and upper should be upper > lower >= 0.')
    return image

def RandomGamma(image, gamma, seed=None):
    assert isinstance(gamma, (int, float, list, tuple)), 'delta should be one of int, float, list, tuple.'
    if isinstance(gamma, (int, float)):
        if 0<=gamma:
            image = tf.image.adjust_gamma(image, gamma, gain=1)
        else:
            raise ValueError('if gamma type one of int or float, must be gamma >= 0.')
    elif 0<=gamma[0]<gamma[1]:
        random_gamma = tf.random.uniform([], gamma[0], gamma[1], seed=seed)
        image = tf.image.adjust_gamma(image, random_gamma, gain=1)
    else:
        raise ValueError('if gamma type one of tuple or list, lower and upper should be upper > lower >= -1.')
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

def RandomRotation(image, k=[0, 1, 2, 3], random=True, seed=None):
    assert isinstance(random, bool), 'random should be bool type.'
    if random:
        assert isinstance(k, list), 'if random is True, sublist in the [0, 1, 2, 3].'
        k_value = tf.convert_to_tensor(k)
        index = tf.argmax(tf.random.uniform([tf.shape(k_value)[0]], 0, 1))
        image = tf.image.rot90(image, k=k_value[index])
    else:
        assert k in [1, 2, 3], 'if random is False, should be int one of [1, 2, 3].'
        image = tf.image.rot90(image, k)
    return image
