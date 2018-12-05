import tensorflow as tf

inception3_url = None

def _make_branch(tensor, use_pool, *conv_settings):
    if use_pool == 'avg':
        x = tf.keras.layers.ZeroPadding2D(((1,1), (1,1)))(tensor)
        x = tf.keras.layers.AvgPool2D(3, 1)(x)
    elif use_pool == 'max':
        x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2)(tensor)
    for setting in conv_settings:
        if setting[3] is not None:
            x = tf.keras.layers.ZeroPadding2D(setting[3])(x)
        x = tf.keras.layers.Conv2D(setting[0], setting[1], setting[2], use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
    return x

def _make_A(tensor, pool_features):
    a1 = _make_branch(tensor, None, (64, 1, 1, None))
    a2 = _make_branch(tensor, None, (48, 1, 1, None), (64, 5, 1, 2))
    a3 = _make_branch(tensor, None, (64, 1, 1, None), (96, 3, 1, 1), (96, 3, 1, 1))
    a4 = _make_branch(tensor, 'avg', (pool_features, 1, 1, None))
    x = tf.keras.layers.Concatenate()([a1, a2, a3, a4])
    return x

def _make_B(tensor):
    a1 = _make_branch(tensor, None, (384, 3, 2, None))
    a2 = _make_branch(tensor, None, (64, 1, 1, None), (96, 3, 1, 1), (96, 3, 2, None))
    a3 = _make_branch(tensor, 'max')
    x = tf.keras.layers.Concatenate()([a1, a2, a3])
    return x

def _make_C(tensor, channels_7x7):
    a1 = _make_branch(tensor, None, (192, 1, 1, None))
    a2 = _make_branch(tensor, None, (channels_7x7, 1, 1, None), (channels_7x7, (1, 7), 1, (0, 3)), (192, (7, 1), 1, (3, 0)))
    a3 = _make_branch(tensor, None, (channels_7x7, 1, 1, None), (channels_7x7, (7, 1), 1, (3, 0)), (channels_7x7, (1, 7), 1, (0, 3)),
                      (channels_7x7, (7, 1), 1, (3, 0)), (192, (1, 7), 1, (0, 3)))
    a4 = _make_branch(tensor, 'avg', (192, 1, 1, None))
    x = tf.keras.layers.Concatenate()([a1, a2, a3, a4])
    return x

def _make_D(tensor):
    a1 = _make_branch(tensor, None, (192, 1, 1, None), (320, 3, 2, None))
    a2 = _make_branch(tensor, None, (192, 1, 1, None), (192, (1, 7), 1, (0, 3)), (192, (7, 1), 1, (3, 0)), (192, 3, 2, None))
    a3 = _make_branch(tensor, 'max')
    x = tf.keras.layers.Concatenate()([a1, a2, a3])
    return x


def _make_E(tensor):
    a1 = _make_branch(tensor, None, (320, 1, 1, None))
    a2 = _make_branch(tensor, None, (384, 1, 1, None))
    a3 = _make_branch(a2, None, (384, (1, 3), 1, (0, 1)))
    a4 = _make_branch(a2, None, (384, (3, 1), 1, (1, 0)))
    a5 = tf.keras.layers.Concatenate()([a3, a4])
    a6 = _make_branch(tensor, None, (448, 1, 1, None), (384, 3, 1, 1))
    a7 = _make_branch(a6, None, (384, (1, 3), 1, (0, 1)))
    a8 = _make_branch(a6, None, (384, (3, 1), 1, (1, 0)))
    a9 = tf.keras.layers.Concatenate()([a7, a8])
    a10 = _make_branch(tensor, 'avg', (192, 1, 1, None))
    x = tf.keras.layers.Concatenate()([a1, a5, a9, a10])
    return x

def Inception3(tensor):
    x = tf.keras.layers.Conv2D(32, 3, 2, use_bias=False)(tensor)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(32, 3, 1, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.ZeroPadding2D(((1,1), (1,1)))(x)
    x = tf.keras.layers.Conv2D(64, 3, 1, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool2D(3, 2)(x)
    x = tf.keras.layers.Conv2D(80, 1, 1, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(192, 3, 1, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool2D(3, 2)(x)
    x = _make_A(x, 32)
    x = _make_A(x, 64)
    x = _make_A(x, 64)
    x = _make_B(x)
    x = _make_C(x, 128)
    x = _make_C(x, 160)
    x = _make_C(x, 160)
    x = _make_C(x, 192)
    x = _make_D(x)
    x = _make_E(x)
    x = _make_E(x)
    x = tf.keras.layers.AvgPool2D(8)(x)
    return x

def inception_v3(input_shape, include_top=True, pretrain_file=False, classes=1000):
    if isinstance(pretrain_file, str) and include_top and classes != 1000:
        raise ValueError('If using `pretrain weights` with `include_top` as true, `classes` should be 1000')
    image = tf.keras.Input(shape=input_shape)
    x = Inception3(image)
    if include_top:
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(classes)(x)
    model = tf.keras.Model(image, x)
    if isinstance(pretrain_file, str):
        if tf.gfile.Exists(pretrain_file):
            model.load_weights(pretrain_file, by_name=True)
        else:
            tf.gfile.MakeDirs(pretrain_file)
            tf.gfile.DeleteRecursively(pretrain_file)
            tf.keras.utils.get_file(pretrain_file, inception3_url)
            model.load_weights(pretrain_file, by_name=True)
    return model
