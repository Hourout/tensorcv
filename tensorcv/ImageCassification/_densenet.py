import tensorflow as tf

densenet_url = {'densenet121': None,
                'densenet161': None,
                'densenet169': None,
                'densenet201': None}

densenet_spec = {121: (64, 32, [6, 12, 24, 16]),
                 161: (96, 48, [6, 12, 36, 24]),
                 169: (64, 32, [6, 12, 32, 32]),
                 201: (64, 32, [6, 12, 48, 32])}

def DenseNet(tensor, num_init_features, growth_rate, block_config):
    x = tf.keras.layers.ZeroPadding2D(padding=((3, 3), (3, 3)))(tensor)
    x = tf.keras.layers.Conv2D(num_init_features, 7, 2, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x = tf.keras.layers.MaxPool2D(3, 2)(x)
    num_features = num_init_features
    for i, num_layers in enumerate(block_config):
        for j in range(num_layers):
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)
            x = tf.keras.layers.Conv2D(bn_size * growth_rate, 1, use_bias=False)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)
            x = tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
            x = tf.keras.layers.Conv2D(growth_rate, 3, use_bias=False)(x)
        num_features = num_features + num_layers * growth_rate
        if i != len(block_config) - 1:
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)
            x = tf.keras.layers.Conv2D(num_features // 2, 1, use_bias=False)(x)
            x = tf.keras.layers.AvgPool2D(2, 2)(x)
            num_features = num_features // 2
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.AvgPool2D(7)(x)
    return x

def get_densenet(num_layers, mode, input_shape, include_top, pretrain_file, classes):
    if isinstance(pretrain_file, str) and include_top and classes != 1000:
        raise ValueError('If using `pretrain weights` with `include_top` as true, `classes` should be 1000')
    num_init_features, growth_rate, block_config = densenet_spec[num_layers]
    image = tf.keras.Input(shape=input_shape)
    x = DenseNet(image, num_init_features, growth_rate, block_config)
    if include_top:
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(classes)(x)
    model = tf.keras.Model(image, x, name=mode)
    if isinstance(pretrain_file, str):
        if tf.gfile.Exists(pretrain_file):
            model.load_weights(pretrain_file, by_name=True)
        else:
            tf.gfile.MakeDirs(pretrain_file)
            tf.gfile.DeleteRecursively(pretrain_file)
            tf.keras.utils.get_file(pretrain_file, densenet_url(mode))
            model.load_weights(pretrain_file, by_name=True)
    return model

def densenet121(input_shape, include_top=True, pretrain_file=False, classes=1000):
    return get_densenet(121, 'densenet121', input_shape, include_top, pretrain_file, classes)

def densenet161(input_shape, include_top=True, pretrain_file=False, classes=1000):
    return get_densenet(161, 'densenet161', input_shape, include_top, pretrain_file, classes)

def densenet169(input_shape, include_top=True, pretrain_file=False, classes=1000):
    return get_densenet(169, 'densenet169', input_shape, include_top, pretrain_file, classes)

def densenet201(input_shape, include_top=True, pretrain_file=False, classes=1000):
    return get_densenet(201, 'densenet201', input_shape, include_top, pretrain_file, classes)
