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

def RandomCentralCropResize(image, central_rate, size, method=0, seed=None):
    assert isinstance(central_rate, (int, float, tuple, list)), 'central_rate should be one of int, float, tuple, list.'
    assert isinstance(size, (tuple, list)), 'size should be one of tuple, list.'
    assert method in [0, 1, 2, 3], 'method should be one of "0:bilinear", "1:nearest_neighbor", "2:bicubic", "3:area"'
    if isinstance(central_rate, (int, float)):
        if 0<central_rate<=1:
            image = tf.image.central_crop(image, central_fraction=central_rate)
        else:
            raise ValueError('if central_rate type one of int or float, must be in the interval (0, 1].')
    elif 0<central_rate[0]<central_rate[1]<=1:
        random_central_rate = tf.random.uniform([], central_rate[0], central_rate[1], seed=seed)
        image = tf.image.central_crop(image, central_fraction=random_central_rate)
    else:
        raise ValueError('if central_rate type one of tuple or list, lower and upper should be 1 >= upper > lower > 0.')
    image = tf.image.resize_images(image, size=size, method=method)
    return image

def RandomPointCropResize(image, height_rate, width_rate, size, method=0, seed=None):
    assert isinstance(height_rate, (int, float)), 'height_rate should be one of int, float.'
    assert isinstance(width_rate, (int, float)), 'width_rate should be one of int, float.'
    assert isinstance(size, (tuple, list)), 'size should be one of tuple, list.'
    assert method in [0, 1, 2, 3], 'method should be one of "0:bilinear", "1:nearest_neighbor", "2:bicubic", "3:area"'
    image = tf.convert_to_tensor(image)
    shape = image.get_shape().as_list()
    if 0<height_rate<=1 and 0<width_rate<=1:
        offset_height = tf.math.multiply(tf.random.uniform([], 0, 1-height_rate, seed=seed), shape[-3])
        offset_width = tf.math.multiply(tf.random.uniform([], 0, 1-width_rate, seed=seed), shape[-2])
        target_height = tf.math.multiply(height_rate, shape[-3])
        target_width = tf.math.multiply(width_rate, shape[-2])
        image = tf.image.crop_to_bounding_box(image, offset_height, offset_width, target_height, target_width)
    else:
        raise ValueError('height_rate and width_rate should be in the interval (0, 1].')
    image = tf.image.resize_images(image, size=size, method=method)
    return image

def Normalize(image, mean=None, std=None):
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    assert image.get_shape().ndims==3, 'image ndims must be 3.'
    if mean is None and std is None:
        image = tf.image.per_image_standardization(image)
    else:
        assert isinstance(mean, (int, float, tuple, list)), 'mean type one of int, float, tuple, list.'
        assert isinstance(std, (int, float, tuple, list)), 'std type one of int, float, tuple, list.'
        image = tf.math.divide(tf.math.subtract(image, mean), std)
    return image

def RandomGaussianNoise(image, scale=1, mean=0.0, std=1.0, seed=None):
    assert isinstance(scale, (int, float)), 'scale type should be one of int, float.'
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image_shape = image.get_shape().as_list()
    if isinstance(mean, (int, float)):
        if isinstance(std, (int, float)):
            image = tf.math.add(tf.math.multiply(tf.random.normal(image_shape, mean, std, seed=seed), scale), image)
        elif isinstance(std, (tuple, list)):
            random_std = tf.random.uniform([], std[0], std[1])
            image = tf.math.add(tf.math.multiply(tf.random.normal(image_shape, mean, random_std, seed=seed), scale), image)
        else:
            raise ValueError('std type should be one of int, float, tuple, list.')
    elif isinstance(mean, (tuple, list)):
        if isinstance(std, (int, float)):
            random_mean = tf.random.uniform([], mean[0], mean[1])
            image = tf.math.add(tf.math.multiply(tf.random.normal(image_shape, random_mean, std, seed=seed), scale), image)
        elif isinstance(std, (tuple, list)):
            random_mean = tf.random.uniform([], mean[0], mean[1])
            random_std = tf.random.uniform([], std[0], std[1])
            image = tf.math.add(tf.math.multiply(tf.random.normal(image_shape, random_mean, random_std, seed=seed), scale), image)
        else:
            raise ValueError('std type should be one of int, float, tuple, list.')
    else:
        raise ValueError('mean type should be one of int, float, tuple, list.')
