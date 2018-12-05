import tensorflow as tf


def BasicBlockV1(tensor, channels, stride, downsample=False, last_gamma=False, use_se=False):
    x = tf.keras.layers.ZeroPadding2D(1)(tensor)
    x = tf.keras.layers.Conv2D(channels, 3, stride, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(body)
    x = tf.keras.layers.ZeroPadding2D(1)(x)
    x = tf.keras.layers.Conv2D(channels, 3, use_bias=False)(x)
    if not last_gamma:
        x = tf.keras.layers.BatchNormalization()(x)
    else:
        x = tf.keras.layers.BatchNormalization(gamma_initializer='zeros')(x)
    if use_se:
        se = tf.keras.layers.GlobalAvgPool2D()(x)
        se = tf.keras.layers.Dense(channels//16, activation='relu', use_bias=False)(se)
        se = tf.keras.layers.Dense(channels, activation='sigmoid', use_bias=False)(se)
        se = tf.keras.layers.Reshape((1,1,channels))(se)
        se = tf.keras.layers.Lambda(lambda x:tf.broadcast_to(x, [x.shape.as_list()[0], x.shape.as_list()[1], channels]))(se)
        x = tf.keras.layers.Multiply()([se, x])
    if downsample:
        ds = tf.keras.layers.Conv2D(channels, 1, stride, use_bias=False)(tensor)
        ds = tf.keras.layers.BatchNormalization()(ds)
        x = tf.keras.layers.Add()([ds, x])
    else:
        x = tf.keras.layers.Add()([tensor, x])
    x = tf.keras.layers.ReLU()(x)
    return x

def BottleneckV1(tensor, channels, stride, downsample=False, last_gamma=False, use_se=False):
    x = tf.keras.layers.Conv2D(channels//4, 1, stride)(tensor)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.ZeroPadding2D(1)(x)
    x = tf.keras.layers.Conv2D(channels//4, 3, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(channels, 1)(x)
    if not last_gamma:
        x = tf.keras.layers.BatchNormalization()(x)
    else:
        x = tf.keras.layers.BatchNormalization(gamma_initializer='zeros')(x)
    if use_se:
        se = tf.keras.layers.GlobalAvgPool2D()(x)
        se = tf.keras.layers.Dense(channels//16, activation='relu', use_bias=False)(se)
        se = tf.keras.layers.Dense(channels, activation='sigmoid', use_bias=False)(se)
        se = tf.keras.layers.Reshape((1,1,channels))(se)
        se = tf.keras.layers.Lambda(lambda x:tf.broadcast_to(x, [x.shape.as_list()[0], x.shape.as_list()[1], channels]))(se)
        x = tf.keras.layers.Multiply()([se, x])
    if downsample:
        ds = tf.keras.layers.Conv2D(channels, 1, stride, use_bias=False)(tensor)
        ds = tf.keras.layers.BatchNormalization()(ds)
        x = tf.keras.layers.Add()([ds, x])
    else:
        x = tf.keras.layers.Add()([tensor, x])
    x = tf.keras.layers.ReLU()(x)
    return x

def BasicBlockV2(tensor, channels, stride, downsample=False, last_gamma=False, use_se=False):
    x = tf.keras.layers.BatchNormalization()(tensor)
    x = tf.keras.layers.ReLU()(x)
    if downsample:
        ds = tf.keras.layers.Conv2D(channels, 1, stride, use_bias=False)(x)
    x = tf.keras.layers.ZeroPadding2D(1)(x)
    x = tf.keras.layers.Conv2D(channels, 3, stride, use_bias=False)(x)
    if not last_gamma:
        x = tf.keras.layers.BatchNormalization()(x)
    else:
        x = tf.keras.layers.BatchNormalization(gamma_initializer='zeros')(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.ZeroPadding2D(1)(x)
    x = tf.keras.layers.Conv2D(channels, 3, use_bias=False)(x)
    if use_se:
        se = tf.keras.layers.GlobalAvgPool2D()(x)
        se = tf.keras.layers.Dense(channels//16, activation='relu', use_bias=False)(se)
        se = tf.keras.layers.Dense(channels, activation='sigmoid', use_bias=False)(se)
        se = tf.keras.layers.Reshape((1,1,channels))(se)
        se = tf.keras.layers.Lambda(lambda x:tf.broadcast_to(x, [x.shape.as_list()[0], x.shape.as_list()[1], channels]))(se)
        x = tf.keras.layers.Multiply()([se, x])
    if downsample:
        x = tf.keras.layers.Add()([ds, x])
    else:
        x = tf.keras.layers.Add()([tensor, x])
    return x

def BottleneckV2(tensor, channels, stride, downsample=False, last_gamma=False, use_se=False):
    x = tf.keras.layers.BatchNormalization()(tensor)
    x = tf.keras.layers.ReLU()(x)
    if downsample:
        ds = tf.keras.layers.Conv2D(channels, 1, stride, use_bias=False)(x)
    x = tf.keras.layers.Conv2D(channels//4, 1, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.ZeroPadding2D(1)(x)
    x = tf.keras.layers.Conv2D(channels//4, 3, stride, use_bias=False)(x)
    if not last_gamma:
        x = tf.keras.layers.BatchNormalization()(x)
    else:
        x = tf.keras.layers.BatchNormalization(gamma_initializer='zeros')(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(channels, 1, use_bias=False)(x)
    if use_se:
        se = tf.keras.layers.GlobalAvgPool2D()(x)
        se = tf.keras.layers.Dense(channels//16, activation='relu', use_bias=False)(se)
        se = tf.keras.layers.Dense(channels, activation='sigmoid', use_bias=False)(se)
        se = tf.keras.layers.Reshape((1,1,channels))(se)
        se = tf.keras.layers.Lambda(lambda x:tf.broadcast_to(x, [x.shape.as_list()[0], x.shape.as_list()[1], channels]))(se)
        x = tf.keras.layers.Multiply()([se, x])
    if downsample:
        x = tf.keras.layers.Add()([ds, x])
    else:
        x = tf.keras.layers.Add()([tensor, x])
    return x

def ResNetV1(tensor, block, layers, channels, thumbnail=False, last_gamma=False, use_se=False):
    assert len(layers) == len(channels) - 1
    if thumbnail:
        x = tf.keras.layers.ZeroPadding2D(1)(tensor)
        x = tf.keras.layers.Conv2D(channels[0], 3, use_bias=False)(x)
    else:
        x = tf.keras.layers.ZeroPadding2D(3)(tensor)
        x = tf.keras.layers.Conv2D(channels[0], 7, 2, use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPool2D(3, 2)(x)
    for i, num_layer in enumerate(layers):
        stride = 1 if i == 0 else 2
        x = block(x, channels[i+1], stride, channels[i+1]!=channels[i], last_gamma, use_se)
        for _ in range(num_layer-1):
            x = block(x, channels[i+1], 1, False, last_gamma, use_se)
    return x

def ResNetV2(tensor, block, layers, channels, thumbnail=False, last_gamma=False, use_se=False):
    assert len(layers) == len(channels) - 1
    x = tf.keras.layers.BatchNormalization(scale=False, center=False)(tensor)
    if thumbnail:
        x = tf.keras.layers.ZeroPadding2D(1)(x)
        x = tf.keras.layers.Conv2D(channels[0], 3, use_bias=False)(x)
    else:
        x = tf.keras.layers.ZeroPadding2D(3)(x)
        x = tf.keras.layers.Conv2D(channels[0], 7, 2, use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPool2D(3, 2)(x)
    in_channels = channels[0]
    for i, num_layer in enumerate(layers):
        stride = 1 if i == 0 else 2
        x = block(x, channels[i+1], stride, channels[i+1]!=in_channels, last_gamma, use_se)
        for _ in range(num_layer-1):
            x = block(x, channels[i+1], 1, False, last_gamma, use_se)
        in_channels = channels[i+1]
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    return x

resnet_spec = {18: ('basic_block', [2, 2, 2, 2], [64, 64, 128, 256, 512]),
               34: ('basic_block', [3, 4, 6, 3], [64, 64, 128, 256, 512]),
               50: ('bottle_neck', [3, 4, 6, 3], [64, 256, 512, 1024, 2048]),
               101: ('bottle_neck', [3, 4, 23, 3], [64, 256, 512, 1024, 2048]),
               152: ('bottle_neck', [3, 8, 36, 3], [64, 256, 512, 1024, 2048])}

resnet_net_versions = [ResNetV1, ResNetV2]
resnet_block_versions = [{'basic_block': BasicBlockV1, 'bottle_neck': BottleneckV1},
                         {'basic_block': BasicBlockV2, 'bottle_neck': BottleneckV2}]

resnet_url = {'resnet18_v1':None,
              'resnet34_v1':None,
              'resnet50_v1':None,
              'resnet101_v1':None,
              'resnet152_v1':None,
              'resnet18_v2':None,
              'resnet34_v2':None,
              'resnet50_v2':None,
              'resnet101_v2':None,
              'resnet152_v2':None,
              'se_resnet18_v1':None,
              'se_resnet34_v1':None,
              'se_resnet50_v1':None,
              'se_resnet101_v1':None,
              'se_resnet152_v1':None,
              'se_resnet18_v2':None,
              'se_resnet34_v2':None,
              'se_resnet34_v2':None,
              'se_resnet101_v2':None,
              'se_resnet152_v2':None}

def get_resnet(version, num_layers, mode, input_shape, include_top, pretrain_file, classes, use_se):
    assert num_layers in resnet_spec, "Invalid number of layers: %d. Options are %s"%(num_layers, str(resnet_spec.keys()))
    assert 1 <= version <= 2, "Invalid resnet version: %d. Options are 1 and 2."%version
    if isinstance(pretrain_file, str) and include_top and classes != 1000:
        raise ValueError('If using `pretrain weights` with `include_top` as true, `classes` should be 1000')
    block_type, layers, channels = resnet_spec[num_layers]
    resnet_class = resnet_net_versions[version-1]
    block_class = resnet_block_versions[version-1][block_type]
    
    image = tf.keras.Input(shape=input_shape)
    net = resnet_class(image, block_class, layers, channels, use_se=use_se)
    x = tf.keras.layers.GlobalAvgPool2D()(net)
    x = tf.keras.layers.Dense(classes)(x)
    model = tf.keras.Model(image, x, name=mode)
    
    if isinstance(pretrain_file, str):
        if tf.gfile.Exists(pretrain_file):
            model.load_weights(pretrain_file)
        else:
            tf.gfile.MakeDirs(pretrain_file)
            tf.gfile.DeleteRecursively(pretrain_file)
            tf.keras.utils.get_file(pretrain_file, resnet_url(mode))
            model.load_weights(pretrain_file)
    return model

def resnet18_v1(input_shape, include_top=True, pretrain_file=False, classes=1000):
    return get_resnet(1, 18, 'resnet18_v1', input_shape, include_top, pretrain_file, classes, False)

def resnet34_v1(input_shape, include_top=True, pretrain_file=False, classes=1000):
    return get_resnet(1, 34, 'resnet34_v1', input_shape, include_top, pretrain_file, classes, False)

def resnet50_v1(input_shape, include_top=True, pretrain_file=False, classes=1000):
    return get_resnet(1, 50, 'resnet50_v1', input_shape, include_top, pretrain_file, classes, False)

def resnet101_v1(input_shape, include_top=True, pretrain_file=False, classes=1000):
    return get_resnet(1, 101, 'resnet101_v1', input_shape, include_top, pretrain_file, classes, False)

def resnet152_v1(input_shape, include_top=True, pretrain_file=False, classes=1000):
    return get_resnet(1, 152, 'resnet152_v1', input_shape, include_top, pretrain_file, classes, False)

def resnet18_v2(input_shape, include_top=True, pretrain_file=False, classes=1000):
    return get_resnet(2, 18, 'resnet18_v2', input_shape, include_top, pretrain_file, classes, False)

def resnet34_v2(input_shape, include_top=True, pretrain_file=False, classes=1000):
    return get_resnet(2, 34, 'resnet34_v2', input_shape, include_top, pretrain_file, classes, False)

def resnet50_v2(input_shape, include_top=True, pretrain_file=False, classes=1000):
    return get_resnet(2, 50, 'resnet50_v2', input_shape, include_top, pretrain_file, classes, False)

def resnet101_v2(input_shape, include_top=True, pretrain_file=False, classes=1000):
    return get_resnet(2, 101, 'resnet101_v2',input_shape, include_top,  pretrain_file, classes, False)

def resnet152_v2(input_shape, include_top=True, pretrain_file=False, classes=1000):
    return get_resnet(2, 152, 'resnet152_v2', input_shape, include_top, pretrain_file, classes, False)

def se_resnet18_v1(input_shape, include_top=True, pretrain_file=False, classes=1000):
    return get_resnet(1, 18, 'se_resnet18_v1', input_shape, include_top, pretrain_file, classes, True)

def se_resnet34_v1(input_shape, include_top=True, pretrain_file=False, classes=1000):
    return get_resnet(1, 34, 'se_resnet34_v1', input_shape, include_top, pretrain_file, classes, True)

def se_resnet50_v1(input_shape, include_top=True, pretrain_file=False, classes=1000):
    return get_resnet(1, 50, 'se_resnet50_v1', input_shape, include_top, pretrain_file, classes, True)

def se_resnet101_v1(input_shape, include_top=True, pretrain_file=False, classes=1000):
    return get_resnet(1, 101, 'se_resnet101_v1', input_shape, include_top, pretrain_file, classes, True)

def se_resnet152_v1(input_shape, include_top=True, pretrain_file=False, classes=1000):
    return get_resnet(1, 152, 'se_resnet152_v1', input_shape, include_top, pretrain_file, classes, True)

def se_resnet18_v2(input_shape, include_top=True, pretrain_file=False, classes=1000):
    return get_resnet(2, 18, 'se_resnet18_v2', input_shape, include_top, pretrain_file, classes, True)

def se_resnet34_v2(input_shape, include_top=True, pretrain_file=False, classes=1000):
    return get_resnet(2, 34, 'se_resnet34_v2', input_shape, include_top, pretrain_file, classes, True)

def se_resnet50_v2(input_shape, include_top=True, pretrain_file=False, classes=1000):
    return get_resnet(2, 50, 'se_resnet34_v2', input_shape, include_top, pretrain_file, classes, True)

def se_resnet101_v2(input_shape, include_top=True, pretrain_file=False, classes=1000):
    return get_resnet(2, 101, 'se_resnet101_v2', input_shape, include_top, pretrain_file, classes, True)

def se_resnet152_v2(input_shape, include_top=True, pretrain_file=False, classes=1000):
    return get_resnet(2, 152, 'se_resnet152_v2', input_shape, include_top, pretrain_file, classes, True)
