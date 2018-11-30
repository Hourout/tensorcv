import tensorflow as tf


def BasicBlockV1(layer, channels, stride, downsample=False, last_gamma=False, use_se=False):
    body = tf.keras.layers.Conv2D(channels, 3, stride, 'same', use_bias=False)(layer)
    body = tf.keras.layers.BatchNormalization()(body)
    body = tf.keras.layers.ReLU()(body)
    body = tf.keras.layers.Conv2D(channels, 3, stride, 'same', use_bias=False)(body)
    if not last_gamma:
        body = tf.keras.layers.BatchNormalization()(body)
    else:
        body = tf.keras.layers.BatchNormalization(gamma_initializer='zeros')(body)
    if use_se:
        se = tf.keras.layers.GlobalAvgPool2D()(body)
        se = tf.keras.layers.Dense(channels//4, activation='relu', use_bias=False)(se)
        se = tf.keras.layers.Dense(channels*4, activation='sigmoid', use_bias=False)(se)
        se = tf.keras.layers.Reshape((1,1,channels*4))(se)
        se = tf.keras.layers.Lambda(lambda x:tf.broadcast_to(x, [body.shape.as_list()[0], body.shape.as_list()[1], channels*4]))(se)
        body = tf.keras.layers.Multiply()([se, body])
    if downsample:
        layer = tf.keras.layers.Conv2D(channels, 1, stride, 'same', use_bias=False)(layer)
        layer = tf.keras.layers.BatchNormalization()(layer)
    body = tf.keras.layers.Add()([layer, body])
    body = tf.keras.layers.ReLU()(body)
    return body

def BottleneckV1(layer, channels, stride, downsample=False, last_gamma=False, use_se=False):
    body = tf.keras.layers.Conv2D(channels//4, 1, stride, 'same')(layer)
    body = tf.keras.layers.BatchNormalization()(body)
    body = tf.keras.layers.ReLU()(body)
    body = tf.keras.layers.Conv2D(channels//4, 3, 1, 'same')(body)
    body = tf.keras.layers.BatchNormalization()(body)
    body = tf.keras.layers.ReLU()(body)
    body = tf.keras.layers.Conv2D(channels, 1, 1, 'same')(body)
    if not last_gamma:
        body = tf.keras.layers.BatchNormalization()(body)
    else:
        body = tf.keras.layers.BatchNormalization(gamma_initializer='zeros')(body)
    if use_se:
        se = tf.keras.layers.GlobalAvgPool2D()(body)
        se = tf.keras.layers.Dense(channels//4, activation='relu', use_bias=False)(se)
        se = tf.keras.layers.Dense(channels*4, activation='sigmoid', use_bias=False)(se)
        se = tf.keras.layers.Reshape((1,1,channels*4))(se)
        se = tf.keras.layers.Lambda(lambda x:tf.broadcast_to(x, [body.shape.as_list()[0], body.shape.as_list()[1], channels*4]))(se)
        body = tf.keras.layers.Multiply()([se, body])
    if downsample:
        layer = tf.keras.layers.Conv2D(channels, 1, stride, 'same', use_bias=False)(layer)
        layer = tf.keras.layers.BatchNormalization()(layer)
    body = tf.keras.layers.Add()([layer, body])
    body = tf.keras.layers.ReLU()(body)
    return body

def BasicBlockV2(layer, channels, stride, downsample=False, last_gamma=False, use_se=False):
    body = tf.keras.layers.BatchNormalization()(layer)
    body = tf.keras.layers.ReLU()(body)
    if downsample:
        image = tf.keras.layers.Conv2D(channels, 1, stride, 'same', use_bias=False)(body)
    body = tf.keras.layers.Conv2D(channels, 3, stride, 'same')(body)
    if not last_gamma:
        body = tf.keras.layers.BatchNormalization()(body)
    else:
        body = tf.keras.layers.BatchNormalization(gamma_initializer='zeros')(body)
    body = tf.keras.layers.ReLU()(body)
    body = tf.keras.layers.Conv2D(channels, 3, 1, 'same')(body)
    if use_se:
        se = tf.keras.layers.GlobalAvgPool2D()(body)
        se = tf.keras.layers.Dense(channels//4, activation='relu', use_bias=False)(se)
        se = tf.keras.layers.Dense(channels*4, activation='sigmoid', use_bias=False)(se)
        se = tf.keras.layers.Reshape((1,1,channels*4))(se)
        se = tf.keras.layers.Lambda(lambda x:tf.broadcast_to(x, [body.shape.as_list()[0], body.shape.as_list()[1], channels*4]))(se)
        body = tf.keras.layers.Multiply()([se, body])
    body = tf.keras.layers.Add()([layer, body])
    return body

def BottleneckV2(layer, channels, stride, downsample=False, last_gamma=False, use_se=False):
    body = tf.keras.layers.BatchNormalization()(layer)
    body = tf.keras.layers.ReLU()(body)
    if downsample:
        image = tf.keras.layers.Conv2D(channels, 1, stride, 'same', use_bias=False)(body)
    body = tf.keras.layers.Conv2D(channels//4, 1, 1, 'same', use_bias=False)(body)
    body = tf.keras.layers.BatchNormalization()(body)
    body = tf.keras.layers.ReLU()(body)
    body = tf.keras.layers.Conv2D(channels//4, 3, stride, 'same')(body)
    if not last_gamma:
        body = tf.keras.layers.BatchNormalization()(body)
    else:
        body = tf.keras.layers.BatchNormalization(gamma_initializer='zeros')(body)
    body = tf.keras.layers.ReLU()(body)
    body = tf.keras.layers.Conv2D(channels, 1, 1, 'same', use_bias=False)(body)
    if use_se:
        se = tf.keras.layers.GlobalAvgPool2D()(body)
        se = tf.keras.layers.Dense(channels//4, activation='relu', use_bias=False)(se)
        se = tf.keras.layers.Dense(channels*4, activation='sigmoid', use_bias=False)(se)
        se = tf.keras.layers.Reshape((1,1,channels*4))(se)
        se = tf.keras.layers.Lambda(lambda x:tf.broadcast_to(x, [body.shape.as_list()[0], body.shape.as_list()[1], channels*4]))(se)
        body = tf.keras.layers.Multiply()([se, body])
    body = tf.keras.layers.Add()([layer, body])
    return body

def ResNetV1(block, layers, channels, input_shape=None, classes=1000, thumbnail=False, last_gamma=False, use_se=False):
    assert len(layers) == len(channels) - 1
    image = tf.keras.Input(shape=input_shape)
    if thumbnail:
        x = tf.keras.layers.Conv2D(channels[0], 3, 1, 'same')(image)
    else:
        x = tf.keras.layers.Conv2D(channels[0], 7, 2, 'same', use_bias=False)(image)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPool2D(3, 2)(x)
    for i, num_layer in enumerate(layers):
        stride = 1 if i == 0 else 2
        x = block(x, channels[i+1], stride, channels[i+1]!=channels[i], last_gamma, use_se)
        for _ in range(num_layer-1):
            x = block(x, channels[i+1], 1, False, last_gamma, use_se)
    x = tf.keras.layers.GlobalAvgPool2D()(x)
    x = tf.keras.layers.Dense(classes)(x)
    model = tf.keras.Model(image, x)
    return model

def ResNetV2(block, layers, channels, input_shape=None, classes=1000, thumbnail=False, last_gamma=False, use_se=False):
    assert len(layers) == len(channels) - 1
    image = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.BatchNormalization(scale=False, center=False)(image)
    if thumbnail:
        x = tf.keras.layers.Conv2D(channels[0], 3, 1, 'same')(x)
    else:
        x = tf.keras.layers.Conv2D(channels[0], 7, 2, 'same', use_bias=False)(x)
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
    x = tf.keras.layers.GlobalAvgPool2D()(x)
    x = tf.keras.layers.Dense(classes)(x)
    model = tf.keras.Model(image, x)
    return model

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

def get_resnet(version, num_layers, input_shape, mode, pretrain_file=False, classes=1000, use_se=False):
    assert num_layers in resnet_spec, "Invalid number of layers: %d. Options are %s"%(num_layers, str(resnet_spec.keys()))
    assert 1 <= version <= 2, "Invalid resnet version: %d. Options are 1 and 2."%version
    block_type, layers, channels = resnet_spec[num_layers]
    resnet_class = resnet_net_versions[version-1]
    block_class = resnet_block_versions[version-1][block_type]
    net = resnet_class(block_class, layers, channels, input_shape, classes, )
    if isinstance(pretrain_file, str):
        if tf.gfile.Exists(pretrain_file):
            model.load_weights(pretrain_file)
        else:
            tf.gfile.MakeDirs(pretrain_file)
            tf.gfile.DeleteRecursively(pretrain_file)
            tf.keras.utils.get_file(pretrain_file, resnet_url(mode))
            model.load_weights(pretrain_file)
    return 

def resnet18_v1(input_shape, pretrain_file=False, classes=1000):
    return get_resnet(1, 18, input_shape, 'resnet18_v1', pretrain_file, classes, use_se=False)

def resnet34_v1(input_shape, pretrain_file=False, classes=1000):
    return get_resnet(1, 34, input_shape, 'resnet34_v1', pretrain_file, classes, use_se=False)

def resnet50_v1(input_shape, pretrain_file=False, classes=1000):
    return get_resnet(1, 50, input_shape, 'resnet50_v1', pretrain_file, classes, use_se=False)

def resnet101_v1(input_shape, pretrain_file=False, classes=1000):
    return get_resnet(1, 101, input_shape, 'resnet101_v1', pretrain_file, classes, use_se=False)

def resnet152_v1(input_shape, pretrain_file=False, classes=1000):
    return get_resnet(1, 152, input_shape, 'resnet152_v1', pretrain_file, classes, use_se=False)

def resnet18_v2(input_shape, pretrain_file=False, classes=1000):
    return get_resnet(2, 18, input_shape, 'resnet18_v2', pretrain_file, classes, use_se=False)

def resnet34_v2(input_shape, pretrain_file=False, classes=1000):
    return get_resnet(2, 34, input_shape, 'resnet34_v2', pretrain_file, classes, use_se=False)

def resnet50_v2(input_shape, pretrain_file=False, classes=1000):
    return get_resnet(2, 50, input_shape, 'resnet50_v2', pretrain_file, classes, use_se=False)

def resnet101_v2(input_shape, pretrain_file=False, classes=1000):
    return get_resnet(2, 101, input_shape, 'resnet101_v2', pretrain_file, classes, use_se=False)

def resnet152_v2(input_shape, pretrain_file=False, classes=1000):
    return get_resnet(2, 152, input_shape, 'resnet152_v2', pretrain_file, classes, use_se=False)

def se_resnet18_v1(input_shape, pretrain_file=False, classes=1000):
    return get_resnet(1, 18, input_shape, 'se_resnet18_v1', pretrain_file, classes, use_se=True)

def se_resnet34_v1(input_shape, pretrain_file=False, classes=1000):
    return get_resnet(1, 34, input_shape, 'se_resnet34_v1', pretrain_file, classes, use_se=True)

def se_resnet50_v1(input_shape, pretrain_file=False, classes=1000):
    return get_resnet(1, 50, input_shape, 'se_resnet50_v1', pretrain_file, classes, use_se=True)

def se_resnet101_v1(input_shape, pretrain_file=False, classes=1000):
    return get_resnet(1, 101, input_shape, 'se_resnet101_v1', pretrain_file, classes, use_se=True)

def se_resnet152_v1(input_shape, pretrain_file=False, classes=1000):
    return get_resnet(1, 152, input_shape, 'se_resnet152_v1', pretrain_file, classes, use_se=True)

def se_resnet18_v2(input_shape, pretrain_file=False, classes=1000):
    return get_resnet(2, 18, input_shape, 'se_resnet18_v2', pretrain_file, classes, use_se=True)

def se_resnet34_v2(input_shape, pretrain_file=False, classes=1000):
    return get_resnet(2, 34, input_shape, 'se_resnet34_v2', pretrain_file, classes, use_se=True)

def se_resnet50_v2(input_shape, pretrain_file=False, classes=1000):
    return get_resnet(2, 50, input_shape, 'se_resnet34_v2', pretrain_file, classes, use_se=True)

def se_resnet101_v2(input_shape, pretrain_file=False, classes=1000):
    return get_resnet(2, 101, input_shape, 'se_resnet101_v2', pretrain_file, classes, use_se=True)

def se_resnet152_v2(input_shape, pretrain_file=False, classes=1000):
    return get_resnet(2, 152, input_shape, 'se_resnet152_v2', pretrain_file, classes, use_se=True)
