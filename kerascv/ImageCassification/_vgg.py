import tensorflow as tf

vgg_url = {'11': None,
           '13': None,
           '16': None,
           '19': None,
           '11_bn':None,
           '13_bn':None,
           '16_bn':None,
           '19_bn':None}

vgg_spec = {11: ([1, 1, 2, 2, 2], [64, 128, 256, 512, 512]),
            13: ([2, 2, 2, 2, 2], [64, 128, 256, 512, 512]),
            16: ([2, 2, 3, 3, 3], [64, 128, 256, 512, 512]),
            19: ([2, 2, 4, 4, 4], [64, 128, 256, 512, 512])}

def VGG(num_layers, mode, pretrain_file=False, input_shape=None, classes=1000, batch_norm=False):
    layers, filters = vgg_spec[num_layers]
    assert len(layers) == len(filters)
    image = tf.keras.Input(shape=input_shape)
    for i, num in enumerate(layers):
        for j in range(num):
            if (i==0)&(j==0):
                x = tf.keras.layers.Conv2D(filters[i], 3, 1, 'same', kernel_initializer=tf.contrib.layers.xavier_initializer())(image)
            else:
                x = tf.keras.layers.Conv2D(filters[i], 3, 1, 'same', kernel_initializer=tf.contrib.layers.xavier_initializer())(x)
            if batch_norm:
                x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPool2D(2, 2)(x)
    x = tf.keras.layers.Dense(4096, activation='relu', kernel_initializer=tf.initializers.random_normal())(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(4096, activation='relu', kernel_initializer=tf.initializers.random_normal())(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(classes, kernel_initializer=tf.initializers.random_normal())(x)
    model = tf.keras.Model(image, x)
    if isinstance(pretrain_file, str):
        if tf.gfile.Exists(pretrain_file):
            model.load_weights(pretrain_file)
        else:
            tf.gfile.MakeDirs(pretrain_file)
            tf.gfile.DeleteRecursively(pretrain_file)
            tf.keras.utils.get_file(pretrain_file, vgg_url(mode))
            model.load_weights(pretrain_file)
    return model

def vgg11(input_shape, pretrain_file=False, classes=1000):
    return VGG(11, '11', pretrain_file=pretrain_file, input_shape=input_shape, classes=classes, batch_norm=False)

def vgg13(input_shape, pretrain_file=False, classes=1000):
    return VGG(13, '13', pretrain_file=pretrain_file, input_shape=input_shape, classes=classes, batch_norm=False)

def vgg16(input_shape, pretrain_file=False, classes=1000):
    return VGG(16, '16', pretrain_file=pretrain_file, input_shape=input_shape, classes=classes, batch_norm=False)

def vgg19(input_shape, pretrain_file=False, classes=1000):
    return VGG(19, '19', pretrain_file=pretrain_file, input_shape=input_shape, classes=classes, batch_norm=False)

def vgg11_bn(input_shape, pretrain_file=False, classes=1000):
    return VGG(11, '11_bn', pretrain_file=pretrain_file, input_shape=input_shape, classes=classes, batch_norm=True)

def vgg13_bn(input_shape, pretrain_file=False, classes=1000):
    return VGG(13, '13_bn', pretrain_file=pretrain_file, input_shape=input_shape, classes=classes, batch_norm=True)

def vgg16_bn(input_shape, pretrain_file=False, classes=1000):
    return VGG(16, '16_bn', pretrain_file=pretrain_file, input_shape=input_shape, classes=classes, batch_norm=True)

def vgg19_bn(input_shape, pretrain_file=False, classes=1000):
    return VGG(19, '19_bn', pretrain_file=pretrain_file, input_shape=input_shape, classes=classes, batch_norm=True)
