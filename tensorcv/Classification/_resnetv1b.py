import tensorflow as tf

def BasicBlockV1b(tensor, channels, strides, dilation, downsample, previous_dilation, last_gamma, avg_down):
    expansion = 1
    x = tf.keras.layers.ZeroPadding2D(dilation)(tensor)
    x = tf.keras.layers.Conv2D(channels, 3, strides, dilation_rate=dilation, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.ZeroPadding2D(previous_dilation)(x)
    x = tf.keras.layers.Conv2D(channels, 3, dilation_rate=previous_dilation, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    if downsample:
        if avg_down:
            if dilation == 1:
                ds = tf.keras.layers.AvgPool2D(strides, strides)(tensor)
            else:
                ds = tf.keras.layers.AvgPool2D(1, 1)(tensor)
            ds = tf.keras.layers.Conv2D(channels*expansion, 1, use_bias=False)(ds)
            ds = tf.keras.layers.BatchNormalization()(ds)
        else:
            ds = tf.keras.layers.Conv2D(channels*expansion, 1, strides, use_bias=False)(tensor)
            ds = tf.keras.layers.BatchNormalization()(ds)
        x = tf.keras.layers.Add()([ds, x])
    else:
        x = tf.keras.layers.Add()([tensor, x])
    x = tf.keras.layers.ReLU()(x)
    return x

def BottleneckV1b(tensor, channels, strides, dilation, downsample, previous_dilation, last_gamma, avg_down):
    expansion = 4
    x = tf.keras.layers.Conv2D(channels, 1, use_bias=False)(tensor)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.ZeroPadding2D(dilation)(x)
    x = tf.keras.layers.Conv2D(channels, 3, strides, dilation_rate=dilation, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(channel*4, 1, use_bias=False)(x)
    if not last_gamma:
        x = tf.keras.layers.BatchNormalization()(x)
    else:
        x = tf.keras.layers.BatchNormalization(gamma_initializer='zeros')(x)
    if downsample:
        if avg_down:
            if dilation == 1:
                ds = tf.keras.layers.AvgPool2D(strides, strides)(tensor)
            else:
                ds = tf.keras.layers.AvgPool2D(1, 1)(tensor)
            ds = tf.keras.layers.nn.Conv2D(channels*expansion, 1, use_bias=False)(ds)
            ds = tf.keras.layers.BatchNormalization()(ds)
        else:
            ds = tf.keras.layers.Conv2D(channels*expansion, 1, strides, use_bias=False)(tensor)
            ds = tf.keras.layers.BatchNormalization()(ds)
        x = tf.keras.layers.Add()([ds, x])
    else:
        x = tf.keras.layers.Add()([tensor, x])
    x = tf.keras.layers.ReLU()(x)
    return x

def ResNetV1b(tensor, block, layers, expansion, dilated=False, last_gamma=False, deep_stem=False, stem_width=32, avg_down=False):
    inplanes = stem_width*2 if deep_stem else 64
    if not deep_stem:
        x = tf.keras.layers.ZeroPadding2D(3)(tensor)
        x = tf.keras.layers.Conv2D(64, 7, 2, use_bias=False)(x)
    else:
        x = tf.keras.layers.ZeroPadding2D(1)(tensor)
        x = tf.keras.layers.Conv2D(stem_width, 3, 2, use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.ZeroPadding2D(1)(x)
        x = tf.keras.layers.Conv2D(stem_width, 3, use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.ZeroPadding2D(1)(x)
        x = tf.keras.layers.Conv2D(stem_width*2, 3, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.ZeroPadding2D(1)(x)
    x = tf.keras.layers.MaxPool2D(3, 2)(x)
    x, inplanes = _make_layer(x, inplanes, block, 64, layers[0], 1, 1, avg_down, last_gamma)
    x, inplanes = _make_layer(x, inplanes, block, 128, layers[1], 2, 1, avg_down, last_gamma)
    if dilated:
        x, inplanes= _make_layer(x, inplanes, block, 256, layers[2], 1, 2, avg_down, last_gamma)
        x, inplanes = _make_layer(x, inplanes, block, 512, layers[3], 1, 4, avg_down, last_gamma)
    else:
        x, inplanes = _make_layer(x, inplanes, block, 256, layers[2], 2, 1, avg_down, last_gamma)
        x, inplanes = _make_layer(x, inplanes, block, 512, layers[3], 2, 1, avg_down, last_gamma)

    def _make_layer(tensor, inplanes, block, channels, blocks, strides=1, dilation=1, avg_down=False, last_gamma=False):
        downsample = strides != 1 or inplanes != channels * expansion
        if dilation in (1, 2):
            x = block(tensor, channels, strides, 1, downsample, dilation, last_gamma, avg_down)
        elif dilation == 4:
            x = block(tensor, channels, strides, 2, downsample, dilation, last_gamma, avg_down)
        else:
            raise RuntimeError("=> unknown dilation size: {}".format(dilation))
        inplanes = channels * expansion
        for i in range(1, blocks):
            x = block(x, channels, 1, dilation, False, dilation, last_gamma)
        return x, inplanes
    return x

def get_resnet_v(mode, input_shape, include_top, pretrain_file, classes, **kwargs):
    if isinstance(pretrain_file, str) and include_top and classes != 1000:
        raise ValueError('If using `pretrain weights` with `include_top` as true, `classes` should be 1000')
    block, expansion = resnet_v_block_versions[mode]
    layers = resnet_v_layer[mode]
    image = tf.keras.Input(shape=input_shape)
    x = ResNetV1b(image, block, layers, expansion, **kwargs)
    if include_top:
        x = tf.keras.layers.GlobalAvgPool2D()(x)
        x = tf.keras.layers.Dense(classes)(x)
    model = tf.keras.Model(image, x, name=mode)
    if isinstance(pretrain_file, str):
        if tf.gfile.Exists(pretrain_file):
            model.load_weights(pretrain_file, by_name=True)
        else:
            tf.gfile.MakeDirs(pretrain_file)
            tf.gfile.DeleteRecursively(pretrain_file)
            tf.keras.utils.get_file(pretrain_file, senet_url(mode))
            model.load_weights(pretrain_file, by_name=True)
    return model

resnet_v_url = {'resnet18_v1b':None,
                'resnet34_v1b':None,
                'resnet50_v1b':None,
                'resnet101_v1b':None,
                'resnet152_v1b':None,
                'resnet50_v1c':None,
                'resnet101_v1c':None,
                'resnet152_v1c':None,
                'resnet50_v1d':None,
                'resnet101_v1d':None,
                'resnet152_v1d':None,
                'resnet50_v1e':None,
                'resnet101_v1e':None,
                'resnet152_v1e':None,
                'resnet50_v1s':None,
                'resnet101_v1s':None,
                'resnet152_v1s':None}

resnet_v_block_versions = {'resnet18_v1b': (BasicBlockV1b, 1),
                           'resnet34_v1b': (BasicBlockV1b, 1),
                           'resnet50_v1b': (BottleneckV1b, 4),
                           'resnet101_v1b': (BottleneckV1b, 4),
                           'resnet152_v1b': (BottleneckV1b, 4),
                           'resnet50_v1c': (BottleneckV1b, 4),
                           'resnet101_v1c': (BottleneckV1b, 4),
                           'resnet152_v1c': (BottleneckV1b, 4),
                           'resnet50_v1d': (BottleneckV1b, 4),
                           'resnet101_v1d': (BottleneckV1b, 4),
                           'resnet152_v1d': (BottleneckV1b, 4),
                           'resnet50_v1e': (BottleneckV1b, 4),
                           'resnet101_v1e': (BottleneckV1b, 4),
                           'resnet152_v1e': (BottleneckV1b, 4),
                           'resnet50_v1s': (BottleneckV1b, 4),
                           'resnet101_v1s': (BottleneckV1b, 4),
                           'resnet152_v1s': (BottleneckV1b, 4)}

resnet_v_layer = {'resnet18_v1b': [2, 2, 2, 2],
                  'resnet34_v1b': [3, 4, 6, 3],
                  'resnet50_v1b': [3, 4, 6, 3],
                  'resnet101_v1b': [3, 4, 23, 3],
                  'resnet152_v1b': [3, 8, 36, 3],
                  'resnet50_v1c': [3, 4, 6, 3],
                  'resnet101_v1c': [3, 4, 23, 3],
                  'resnet152_v1c': [3, 8, 36, 3],
                  'resnet50_v1d': [3, 4, 6, 3],
                  'resnet101_v1d': [3, 4, 23, 3],
                  'resnet152_v1d': [3, 8, 36, 3],
                  'resnet50_v1e': [3, 4, 6, 3],
                  'resnet101_v1e': [3, 4, 23, 3],
                  'resnet152_v1e': [3, 8, 36, 3],
                  'resnet50_v1s': [3, 4, 6, 3],
                  'resnet101_v1s': [3, 4, 23, 3],
                  'resnet152_v1s': [3, 8, 36, 3]}

def resnet18_v1b(input_shape, include_top=True, pretrain_file=False, classes=1000):
    return get_resnet_v('resnet18_v1b', input_shape, include_top, pretrain_file, classes)

def resnet34_v1b(input_shape, include_top=True, pretrain_file=False, classes=1000):
    return get_resnet_v('resnet34_v1b', input_shape, include_top, pretrain_file, classes)

def resnet50_v1b(input_shape, include_top=True, pretrain_file=False, classes=1000):
    return get_resnet_v('resnet50_v1b', input_shape, include_top, pretrain_file, classes)

def resnet101_v1b(input_shape, include_top=True, pretrain_file=False, classes=1000):
    return get_resnet_v('resnet101_v1b', input_shape, include_top, pretrain_file, classes)

def resnet152_v1b(input_shape, include_top=True, pretrain_file=False, classes=1000):
    return get_resnet_v('resnet152_v1b', input_shape, include_top, pretrain_file, classes)

def resnet50_v1c(input_shape, include_top=True, pretrain_file=False, classes=1000):
    return get_resnet_v('resnet50_v1c', input_shape, include_top, pretrain_file, classes, deep_stem=True)

def resnet101_v1c(input_shape, include_top=True, pretrain_file=False, classes=1000):
    return get_resnet_v('resnet101_v1c', input_shape, include_top, pretrain_file, classes, deep_stem=True)

def resnet152_v1c(input_shape, include_top=True, pretrain_file=False, classes=1000):
    return get_resnet_v('resnet152_v1c', input_shape, include_top, pretrain_file, classes, deep_stem=True)

def resnet50_v1d(input_shape, include_top=True, pretrain_file=False, classes=1000):
    return get_resnet_v('resnet50_v1d', input_shape, include_top, pretrain_file, classes, deep_stem=True, avg_down=True)

def resnet101_v1d(input_shape, include_top=True, pretrain_file=False, classes=1000):
    return get_resnet_v('resnet101_v1d', input_shape, include_top, pretrain_file, classes, deep_stem=True, avg_down=True)

def resnet152_v1d(input_shape, include_top=True, pretrain_file=False, classes=1000):
    return get_resnet_v('resnet152_v1d', input_shape, include_top, pretrain_file, classes, deep_stem=True, avg_down=True)

def resnet50_v1e(input_shape, include_top=True, pretrain_file=False, classes=1000):
    return get_resnet_v('resnet50_v1e', input_shape, include_top, pretrain_file, classes, deep_stem=True, avg_down=True, stem_width=64)

def resnet101_v1e(input_shape, include_top=True, pretrain_file=False, classes=1000):
    return get_resnet_v('resnet101_v1e', input_shape, include_top, pretrain_file, classes, deep_stem=True, avg_down=True, stem_width=64)

def resnet152_v1e(input_shape, include_top=True, pretrain_file=False, classes=1000):
    return get_resnet_v('resnet152_v1e', input_shape, include_top, pretrain_file, classes, deep_stem=True, avg_down=True, stem_width=64)

def resnet50_v1s(input_shape, include_top=True, pretrain_file=False, classes=1000):
    return get_resnet_v('resnet50_v1s', input_shape, include_top, pretrain_file, classes, deep_stem=True, stem_width=64)

def resnet101_v1s(input_shape, include_top=True, pretrain_file=False, classes=1000):
    return get_resnet_v('resnet101_v1s', input_shape, include_top, pretrain_file, classes, deep_stem=True, stem_width=64)

def resnet152_v1s(input_shape, include_top=True, pretrain_file=False, classes=1000):
    return get_resnet_v('resnet152_v1s', input_shape, include_top, pretrain_file, classes, deep_stem=True, stem_width=64)
