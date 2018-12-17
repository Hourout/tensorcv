import tensorflow as tf

def _conv2d(tensor, channel, kernel, padding, stride):
    x = tf.keras.layers.ZeroPadding2D(padding)(tensor)
    x = tf.keras.layers.Conv2D(channel, kernel, stride, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9)(x)
    x = tf.keras.layers.LeakyReLU(0.1)(x)
    return x

def DarknetBasicBlockV3(tensor, channel):
    x = _conv2d(tensor, channel, 1, 0, 1)
    x = _conv2d(x, channel, 3, 1, 1)
    x = tf.keras.layers.Add()([tensor, x])
    return x

def DarknetV3(tensor, layers, channels):
    assert len(layers) == len(channels) - 1, (
            "len(channels) should equal to len(layers) + 1, given {} vs {}".format(
                len(channels), len(layers)))
    x = _conv2d(tensor, channels[0], 3, 1, 1)
    for nlayer, channel in zip(layers, channels[1:]):
        assert channel % 2 == 0, "channel {} cannot be divided by 2".format(channel)
        x = _conv2d(x, channel, 3, 1, 2)
        for _ in range(nlayer):
            x = DarknetBasicBlockV3(x, channel // 2)
    return x


darknet_versions = {'v3': DarknetV3}
darknet_spec = {'v3': {53: ([1, 2, 8, 8, 4], [32, 64, 128, 256, 512, 1024]),}}

def get_darknet(mode, num_layers, input_shape, include_top, pretrain_file, classes):
    specs = darknet_spec[mode]
    assert num_layers in specs, "Invalid number of layers: {}. Options are {}".format(num_layers, str(specs.keys()))
    layers, channels = specs[num_layers]
    darknet_class = darknet_versions[mode]
    image = tf.keras.Input(shape=input_shape)
    x = darknet_class(image, layers, channels)
    if include_top:
        x = tf.keras.layers.GlobalAvgPool2D()(x)
        x = tf.keras.layers.Dense(classes)(x)
    model = tf.keras.Model(image, x)
    if isinstance(pretrain_file, str):
        if tf.gfile.Exists(pretrain_file):
            model.load_weights(pretrain_file, by_name=True)
        else:
            tf.gfile.MakeDirs(pretrain_file)
            tf.gfile.DeleteRecursively(pretrain_file)
            tf.keras.utils.get_file(pretrain_file, alexnet_url)
            model.load_weights(pretrain_file, by_name=True)
    return model

def darknet53(input_shape, include_top=True, pretrain_file=False, classes=1000):
    return get_darknet('v3', 53, input_shape, include_top, pretrain_file, classes)
