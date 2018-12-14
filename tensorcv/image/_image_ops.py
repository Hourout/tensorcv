import tensorflow as tf

def read_image(filename, channel=0, image_format='mix'):
    """Convenience function for read image type one of `bmp`, `gif`, `jpeg`, `jpg`, and `png`.
    
    Detects whether an image is a BMP, GIF, JPEG, JPG, or PNG, and performs the
    appropriate operation to convert the input bytes `string` into a `Tensor`
    of type `dtype`.
    
    Note: `gif` returns a 4-D array `[num_frames, height, width, 3]`, as
    opposed to `bmp`, `jpeg`, `jpg` and `png`, which return 3-D
    arrays `[height, width, num_channels]`. Make sure to take this into account
    when constructing your graph if you are intermixing GIF files with BMP, JPEG, JPG,
    and/or PNG files.
    Args:
        filename: 0-D `string`. image absolute path.
        channels: An optional `int`. Defaults to `0`. Number of color channels for
                  the decoded image. 1 for `grayscale` and 3 for `rgb`.
        image_format: 0-D `string`. image format type one of `bmp`, `gif`, `jpeg`,
                      `jpg`, `png` and `mix`. `mix` mean contains many types image format.
    Returns:
        `Tensor` with type uint8 and shape `[height, width, num_channels]` for
        BMP, JPEG, and PNG images and shape `[num_frames, height, width, 3]` for
        GIF images.
    Raises:
        ValueError: On incorrect number of channels.
    """
    assert channel in [0, 1, 3], 'channel should be one of [0, 1, 3].'
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
    """Adjust the brightness of RGB or Grayscale images.
    
    Tips:
        delta extreme value in the interval [-1, 1], >1 to white, <-1 to black.
        a suitable interval is [-0.5, 0.5].
        0 means pixel value no change.
    Args:
        image: Tensor or array. An image.
        delta: if int, float, Amount to add to the pixel values.
               if list, tuple, randomly picked in the interval
               `[delta[0], delta[1])` to add to the pixel values.
        seed: A Python integer. Used to create a random seed. See
             `tf.set_random_seed` for behavior.
    Returns:
        A brightness-adjusted tensor of the same shape and type as `image`.
    Raises:
        ValueError: if `delta` type is error.
    """
    if isinstance(delta, (int, float)):
        assert -1<=delta<=1, 'delta should be in the interval [-1, 1].'
        image = tf.image.adjust_brightness(image, delta)
    elif isinstance(delta, (list, tuple)):
        assert -1<=delta[0]<delta[1]<=1, 'delta should be 1 >= delta[1] > delta[0] >= -1.'
        random_delta = tf.random.uniform([], delta[0], delta[1], seed=seed)
        image = tf.image.adjust_brightness(image, random_delta)
    else:
        raise ValueError('delta should be one of int, float, list, tuple.')
    return image

def RandomContrast(image, delta, seed=None):
    """Adjust contrast of RGB or grayscale images.
    
    `images` is a tensor of at least 3 dimensions.  The last 3 dimensions are
    interpreted as `[height, width, channels]`.  The other dimensions only
    represent a collection of images, such as `[batch, height, width, channels].`
  
    Contrast is adjusted independently for each channel of each image.
    
    For each channel, this Ops computes the mean of the image pixels in the
    channel and then adjusts each component `x` of each pixel to
    `(x - mean) * delta + mean`.
    
    Tips:
        1 means pixel value no change.
        0 means all pixel equal. 
        a suitable interval is (0, 4].
    Args:
        images: Tensor or array. An image. At least 3-D.
        delta: if int, float, a float multiplier for adjusting contrast.
               if list, tuple, randomly picked in the interval
               `[delta[0], delta[1])` , value is float multiplier for adjusting contrast.
        seed: A Python integer. Used to create a random seed. See
             `tf.set_random_seed` for behavior.
    Returns:
        The contrast-adjusted image or images tensor of the same shape and type as `image`.
    Raises:
        ValueError: if `delta` type is error.
    """
    if isinstance(delta, (int, float)):
        image = tf.image.adjust_contrast(image, delta)
    elif isinstance(delta, (list, tuple)):
        assert delta[0]<delta[1], 'delta should be delta[1] > delta[0].'
        random_delta = tf.random.uniform([], delta[0], delta[1], seed=seed)
        image = tf.image.adjust_contrast(image, random_delta)
    else:
        raise ValueError('delta should be one of int, float, list, tuple.')
    return image

def RandomHue(image, delta, seed=None):
    """Adjust hue of an RGB image.
    
    `image` is an RGB image.  The image hue is adjusted by converting the
    image to HSV and rotating the hue channel (H) by `delta`.
    The image is then converted back to RGB.
    
    Tips:
        `delta` should be in the interval `[-1, 1]`, but any value is allowed.
        a suitable interval is [-0.5, 0.5].
        int value means pixel value no change.
    Args:
        image: Tensor or array. RGB image or images. Size of the last dimension must be 3.
        delta: if float, How much to add to the hue channel.
               if list, tuple, randomly picked in the interval
               `[delta[0], delta[1])` , value is how much to add to the hue channel.
        seed: A Python integer. Used to create a random seed. See
             `tf.set_random_seed` for behavior.
    Returns:
        The hue-adjusted image or images tensor of the same shape and type as `image`.
    Raises:
        ValueError: if `delta` type is error.
    """
    if isinstance(delta, (int, float)):
        image = tf.image.adjust_hue(image, delta)
    elif isinstance(delta, (list, tuple)):
        assert delta[0]<delta[1], 'delta should be delta[1] > delta[0].'
        random_delta = tf.random.uniform([], delta[0], delta[1], seed=seed)
        image = tf.image.adjust_hue(image, random_delta)
    else:
        raise ValueError('delta should be one of int, float, list, tuple.')
    return image

def RandomSaturation(image, delta, seed=None):
    """Adjust saturation of an RGB image.
    
    `image` is an RGB image.  The image saturation is adjusted by converting the
    image to HSV and multiplying the saturation (S) channel by `delta` and clipping.
    The image is then converted back to RGB.
    
    Tips:
        if delta <= 0, image channels value are equal, image color is gray.
        a suitable interval is delta >0
    Args:
        image: RGB image or images. Size of the last dimension must be 3.
        delta: if int, float, Factor to multiply the saturation by.
               if list, tuple, randomly picked in the interval
               `[delta[0], delta[1])` , value is factor to multiply the saturation by.
        seed: A Python integer. Used to create a random seed. See
             `tf.set_random_seed` for behavior.
    Returns:
        The saturation-adjusted image or images tensor of the same shape and type as `image`.
    Raises:
        ValueError: if `delta` type is error.
    """
    if isinstance(delta, (int, float)):
        image = tf.image.adjust_saturation(image, delta)
    elif isinstance(delta, (list, tuple)):
        assert delta[0]<delta[1], 'delta should be delta[1] > delta[0].'
        image = tf.image.random_saturation(image, delta[0], delta[1], seed=seed)
    else:
        raise ValueError('delta should be one of int, float, list, tuple.')
    return image

def RandomGamma(image, gamma, seed=None):
    """Performs Gamma Correction on the input image.
    
    Also known as Power Law Transform. This function transforms the
    input image pixelwise according to the equation `Out = In**gamma`
    after scaling each pixel to the range 0 to 1.
    
    Tips:
        For gamma greater than 1, the histogram will shift towards left and
        the output image will be darker than the input image.
        For gamma less than 1, the histogram will shift towards right and
        the output image will be brighter than the input image.
        if gamma is 1, image pixel value no change.
    Args:
        image : A Tensor.
        gamma : if int, float, Non negative real number.
                if list, tuple, randomly picked in the interval
                `[delta[0], delta[1])` , value is Non negative real number.
        seed: A Python integer. Used to create a random seed. See
              `tf.set_random_seed` for behavior.
    Returns:
        A float Tensor. Gamma corrected output image.
    Raises:
        ValueError: If gamma is negative.
    References:
        [1] http://en.wikipedia.org/wiki/Gamma_correction
    """
    image = tf.cast(image, dtype=tf.float32)
    if isinstance(gamma, (int, float)):
        assert 0<gamma, 'gamma should be > 0.'
        image = tf.image.adjust_gamma(image, gamma, gain=1)
    elif isinstance(gamma, (list, tuple)):
        assert 0<gamma[0]<gamma[1], 'gamma should be gamma[1] > gamma[0] > 0.'
        random_gamma = tf.random.uniform([], gamma[0], gamma[1], seed=seed)
        image = tf.image.adjust_gamma(image, random_gamma, gain=1)
    else:
        raise ValueError('gamma should be one of int, float, list, tuple.')
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
    return image

def RandomNoisePoisson(image, scale=1, lam=1.0, seed=None):
    assert isinstance(scale, (int, float)), 'scale type should be one of int, float.'
    image = tf.cast(image, dtype=tf.float32)
    image_shape = image.get_shape().as_list()
    if isinstance(lam, (int, float)):
        image = tf.math.add(tf.math.multiply(tf.random.poisson(lam, image_shape, seed=seed), scale), image)
    elif isinstance(lam, (tuple, list)):
        random_lam = tf.random.uniform([], lam[0], lam[1])
        image = tf.math.add(tf.math.multiply(tf.random.poisson(random_lam, image_shape, seed=seed), scale), image)
    else:
        raise ValueError('lam type should be one of int, float, tuple, list.')
    return image

def RandomNoiseMask(image, keep_prob=0.95, seed=None):
    if isinstance(keep_prob, float):
        mask = tf.clip_by_value(tf.nn.dropout(tf.random.uniform(image_shape, 1., 2.), keep_prob), 0., 1.)
        image = tf.math.multiply(mask, image)
    elif isinstance(keep_prob, (tuple, list)):
        random_keep_prob = tf.random.uniform([], keep_prob[0], keep_prob[1], seed=seed)
        mask = tf.clip_by_value(tf.nn.dropout(tf.random.uniform(image_shape, 1., 2.), random_keep_prob), 0., 1.)
        image = tf.math.multiply(mask, image)
    else:
        raise ValueError('keep_prob type should be one of float, tuple, list.')
    return image

def RandomNoiseSaltPepper(image, keep_prob=0.8, seed=None):
    assert isinstance(keep_prob, (int, float)), 'keep_prob type should be one of int, float.'
    image = tf.cast(image, dtype=tf.float32)
    image_shape = image.get_shape().as_list()
    if isinstance(keep_prob, (int, float)):
        noise = tf.clip_by_value(tf.math.floor(tf.random.uniform(image_shape, 0.5-0.5/keep_prob, 0.5+0.5/keep_prob)), -1, 1)
        image = tf.clip_by_value(tf.math.add(tf.math.multiply(noise, 255.), image), 0, 255)
    elif isinstance(keep_prob, (tuple, list)):
        random_keep_prob = tf.random.uniform([], keep_prob[0], keep_prob[1], seed=seed)
        noise = tf.clip_by_value(tf.math.floor(tf.random.uniform(image_shape, 0.5-0.5/keep_prob, 0.5+0.5/keep_prob)), -1, 1)
        image = tf.clip_by_value(tf.math.add(tf.math.multiply(noise, 255.), image), 0, 255)
    else:
        raise ValueError('keep_prob type should be one of int, float, tuple, list.')
    return image

def RandomRescale(image, scale, seed=None):
    if isinstance(scale, (int, float)):
        image = tf.math.multiply(image, scale)
    elif isinstance(scale, (tuple, list)):
        random_scale = tf.random.uniform([], scale[0], scale[1], seed=seed)
        image = tf.math.multiply(image, random_scale)
    else:
        raise ValueError('scale type should be one of int, float, tuple, list.')
    return image
