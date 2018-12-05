import math
import tensorflow as tf

def SEBlock(tensor, channels, cardinality, bottleneck_width, stride, downsample, downsample_kernel_size):
    group_width = cardinality * int(math.floor(channels * (bottleneck_width / 64)))
    x = tf.keras.layers.Conv2D(group_width//2, 1, use_bias=False)(tensor)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.ZeroPadding2D(((1, 1), (1, 1)))(x)
    x = tf.keras.layers.Conv2D(group_width, 3, stride, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(channels*4, 1, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    se = tf.keras.layers.GlobalAvgPool2D()(x)
    se = tf.keras.layers.Conv2D(channels//4, 1, activation='relu')(se)
    se = tf.keras.layers.Conv2D(channels*4, 1, activation='sigmoid')(se)
    se = tf.keras.layers.Reshape((1,1,channels*4))(se)
    se = tf.keras.layers.Lambda(lambda x:tf.broadcast_to(x, [x.shape.as_list()[0], x.shape.as_list()[1], channels*4]))(se)
    x = tf.keras.layers.Multiply()([se, x])
    if downsample:
        if downsample_kernel_size == 3:
            ds = tf.keras.layers.ZeroPadding2D(((1, 1), (1, 1)))(tensor)
            ds = tf.keras.layers.Conv2D(channels*4, downsample_kernel_size, stride, use_bias=False)(ds)
        else:
            ds = tf.keras.layers.Conv2D(channels*4, downsample_kernel_size, stride, use_bias=False)(tensor)
        ds = tf.keras.layers.BatchNormalization()(ds)
        x = tf.keras.layers.Add()([x, ds])
    else:
        x = tf.keras.layers.Add()([x, tensor])
    x = tf.keras.layers.ReLU()(x)
    return x

def SENet(tensor, layers, cardinality, bottleneck_width, channels=64):
    x = tf.keras.layers.ZeroPadding2D(((1, 1), (1, 1)))(tensor)
    x = tf.keras.layers.Conv2D(channels, 3, 2, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.ZeroPadding2D(((1, 1), (1, 1)))(x)
    x = tf.keras.layers.Conv2D(channels, 3, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.ZeroPadding2D(((1, 1), (1, 1)))(x)
    x = tf.keras.layers.Conv2D(channels*2, 3, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool2D(3, 2)(x)
    for i, num_layer in enumerate(layers):
        stride = 1 if i == 0 else 2
        downsample_kernel_size = 1 if i==0 else 3
        x = SEBlock(x, channels, cardinality, bottleneck_width, stride, True, downsample_kernel_size)
        for _ in range(num_layers-1):
            x = SEBlock(x, channels, cardinality, bottleneck_width, 1, False, 3)
        channels *= 2
    x = tf.keras.layers.GlobalAvgPool2D()(x)
    return x

senet_url = {'senet_50': None;
             'senet_101': None,
             'senet_152': None}

senet_spec = {50: [3, 4, 6, 3],
              101: [3, 4, 23, 3],
              152: [3, 8, 36, 3]}

def get_senet(num_layers, mode, input_shape, include_top, pretrain_file, classes, cardinality=64, bottleneck_width=4):
    if isinstance(pretrain_file, str) and include_top and classes != 1000:
        raise ValueError('If using `pretrain weights` with `include_top` as true, `classes` should be 1000')
    layers = senet_spec[num_layers]
    image = tf.keras.Input(shape=input_shape)
    x = SENet(image, layers, cardinality, bottleneck_width, channels=64)
    if include_top:
        x = tf.keras.layers.Flatten()(x)
        x = rf.keras.layers.Dropout(0.2)(x)
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

def senet_50(input_shape, include_top=True, pretrain_file=False, classes=1000):
    return get_senet(50, 'senet_50', input_shape, include_top, pretrain_file, classes, 64, 4)

def senet_101(input_shape, include_top=True, pretrain_file=False, classes=1000):
    return get_senet(101, 'senet_101', input_shape, include_top, pretrain_file, classes, 64, 4)

def senet_152(input_shape, include_top=True, pretrain_file=False, classes=1000):
    return get_senet(152, 'senet_152', input_shape, include_top, pretrain_file, classes, 64, 4)
