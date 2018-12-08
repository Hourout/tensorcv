import tensorflow as tf

def _make_fire(tensor, squeeze_channels, expand1x1_channels, expand3x3_channels):
    x = tf.keras.layers.Conv2D(squeeze_channels, 1)(tensor)
    x = tf.keras.layers.ReLU()(x)
    x_left = tf.keras.layers.Conv2D(expand1x1_channels, 1)(x)
    x_left = tf.keras.layers.ReLU()(x_left)
    x_right = tf.keras.layers.ZeroPadding2D(1)(x)
    x_right = tf.keras.layers.Conv2D(expand3x3_channels, 3)(x_right)
    x_right = tf.keras.layers.ReLU()(x_right)
    x = tf.keras.layers.Concatenate()([x_left, x_right])
    return x

def SqueezeNet(tensor, mode):
    if mode == 'squeezenet1.0':
        x = tf.keras.layers.Conv2D(96, 7, 2)(tensor)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPool2D(3, 2)(x)
        x = _make_fire(x, 16, 64, 64)
        x = _make_fire(x, 16, 64, 64)
        x = _make_fire(x, 32, 128, 128)
        x = tf.keras.layers.MaxPool2D(3, 2)(x)
        x = _make_fire(x, 32, 128, 128)
        x = _make_fire(x, 48, 192, 192)
        x = _make_fire(x, 48, 192, 192)
        x = _make_fire(x, 64, 256, 256)
        x = tf.keras.layers.MaxPool2D(3, 2)(x)
        x = _make_fire(x, 64, 256, 256)
    else:
        x = tf.keras.layers.Conv2D(64, 3, 2)(tensor)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPool2D(3, 2)(x)
        x = _make_fire(x, 16, 64, 64)
        x = _make_fire(x, 16, 64, 64)
        x = tf.keras.layers.MaxPool2D(3, 2)(x)
        x = _make_fire(x, 32, 128, 128)
        x = _make_fire(x, 32, 128, 128)
        x = tf.keras.layers.MaxPool2D(3, 2)(x)
        x = _make_fire(x, 48, 192, 192)
        x = _make_fire(x, 48, 192, 192)
        x = _make_fire(x, 64, 256, 256)
        x = _make_fire(x, 64, 256, 256)
        x = tf.keras.layers.Dropout(0.5)(x)
    return x

squeezenet_url = {'squeezenet1.0':None,
                  'squeezenet1.1':None}

def get_squeezenet(mode, input_shape, include_top, pretrain_file, classes):
    if isinstance(pretrain_file, str) and include_top and classes != 1000:
        raise ValueError('If using `pretrain weights` with `include_top` as true, `classes` should be 1000')
    image = tf.keras.Input(shape=input_shape)
    x = SqueezeNet(image, mode)
    if include_top:
        x = tf.keras.layers.Conv2D(classes, 1)(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.GlobalAvgPool2D()(x)
    model = tf.keras.Model(image, x, name=mode)
    if isinstance(pretrain_file, str):
        if tf.gfile.Exists(pretrain_file):
            model.load_weights(pretrain_file, by_name=True)
        else:
            tf.gfile.MakeDirs(pretrain_file)
            tf.gfile.DeleteRecursively(pretrain_file)
            tf.keras.utils.get_file(pretrain_file, squeezenet_url(mode))
            model.load_weights(pretrain_file, by_name=True)
    return model

def squeezenet1_0(input_shape, include_top=True, pretrain_file=False, classes=1000):
    return get_squeezenet('squeezenet1.0', input_shape, include_top, pretrain_file, classes)

def squeezenet1_1(input_shape, include_top=True, pretrain_file=False, classes=1000):
    return get_squeezenet('squeezenet1.1', input_shape, include_top, pretrain_file, classes)
