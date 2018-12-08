import tensorflow as tf

def Block(tensor, channels, cardinality, bottleneck_width, stride, downsample, last_gamma, use_se):
    D = int(math.floor(channels * (bottleneck_width / 64)))
    group_width = cardinality * D
    x = tf.keras.layers.Conv2D(group_width, 1, use_bias=False)(tensor)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.ZeroPadding2D(1)(x)
    x = tf.keras.layers.Conv2D(group_width, 3, stride, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(channels * 4, 1, use_bias=False)(x)
    if not last_gamma:
        x = tf.keras.layers.BatchNormalization()(x)
    else:
        x = tf.keras.layers.BatchNormalization(gamma_initializer='zeros')(x)
    if use_se:
        se = tf.keras.layers.GlobalAvgPool2D()(x)
        se = tf.keras.layers.Dense(channels//4, activation='relu', use_bias=False)(se)
        se = tf.keras.layers.Dense(channels*4, activation='sigmoid', use_bias=False)(se)
        se = tf.keras.layers.Reshape((1,1,channels*4))(se)
        se = tf.keras.layers.Lambda(lambda x:tf.broadcast_to(x, [x.shape.as_list()[0], x.shape.as_list()[1], channels*4]))(se)
        x = tf.keras.layers.Multiply()([se, x])
    if downsample:
        ds = tf.keras.layers.Conv2D(channels*4, 1, stride, use_bias=False)(tensor)
        ds = tf.keras.layers.BatchNormalization()(ds)
        x = tf.keras.layers.Add()([ds, x])
    else:
        x = tf.keras.layers.Add()([tensor, x])
    x = tf.keras.layers.ReLU()(x)
    return x

def ResNext(tensor, layers, cardinality, bottleneck_width, last_gamma=False, use_se=False):
    channels = 64
    x = tf.keras.layers.ZeroPadding2D(3)(tensor)
    x = tf.keras.layers.Conv2D(channels, 7, 2, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.ZeroPadding2D(1)(x)
    x = tf.keras.layers.MaxPool2D(3, 2)(x)
    for i, num_layer in enumerate(layers):
        stride = 1 if i == 0 else 2
        x = Block(x, channels, cardinality, bottleneck_width, stride, True, last_gamma, use_se)
        for _ in range(num_layer-1):
            x = Block(x, channels, cardinality, bottleneck_width, 1, False, last_gamma, use_se)
        channels *= 2
    return x

resnext_url = {'resnext50_32x4d':None,
               'resnext101_32x4d':None,
               'resnext101_64x4d':None,
               'se_resnext50_32x4d':None,
               'se_resnext50_32x4d':None,
               'se_resnext101_32x4d':None,
               'se_resnext101_64x4d':None}

resnext_spec = {50: [3, 4, 6, 3],
                101: [3, 4, 23, 3]}

def get_resnext(num_layers, cardinality, bottleneck_width, mode, input_shape, include_top, pretrain_file, classes, use_se):
    if isinstance(pretrain_file, str) and include_top and classes != 1000:
        raise ValueError('If using `pretrain weights` with `include_top` as true, `classes` should be 1000')
    layers = resnext_spec[num_layers]
    image = tf.keras.Input(shape=input_shape)
    x = ResNext(image, layers, cardinality, bottleneck_width, use_se)
    if include_top:
        x = tf.keras.layers.GlobalAvgPool2D()(x)
        x = tf.keras.layers.Dense(classes)(x)
    model = tf.keras.Model(image, x, name=mode)
    
    if isinstance(pretrain_file, str):
        if tf.gfile.Exists(pretrain_file):
            model.load_weights(pretrain_file)
        else:
            tf.gfile.MakeDirs(pretrain_file)
            tf.gfile.DeleteRecursively(pretrain_file)
            tf.keras.utils.get_file(pretrain_file, resnext_url(mode))
            model.load_weights(pretrain_file)
    return model

def resnext50_32x4d(input_shape, include_top=True, pretrain_file=False, classes=1000):
    return get_resnext(50, 32, 4, 'resnext50_32x4d', input_shape, include_top, pretrain_file, classes, False)

def resnext101_32x4d(input_shape, include_top=True, pretrain_file=False, classes=1000):
    return get_resnext(101, 32, 4, 'resnext101_32x4d', input_shape, include_top, pretrain_file, classes, False)

def resnext101_64x4d(input_shape, include_top=True, pretrain_file=False, classes=1000):
    return get_resnext(101, 64, 4, 'resnext101_64x4d', input_shape, include_top, pretrain_file, classes, False)

def se_resnext50_32x4d(input_shape, include_top=True, pretrain_file=False, classes=1000):
    return get_resnext(50, 32, 4, 'se_resnext50_32x4d', input_shape, include_top, pretrain_file, classes, True)

def se_resnext101_32x4d(input_shape, include_top=True, pretrain_file=False, classes=1000):
    return get_resnext(101, 32, 4, 'se_resnext101_32x4d', input_shape, include_top, pretrain_file, classes, True)

def se_resnext101_64x4d(input_shape, include_top=True, pretrain_file=False, classes=1000):
    return get_resnext(101, 64, 4, 'se_resnext101_64x4d', input_shape, include_top, pretrain_file, classes, True)
