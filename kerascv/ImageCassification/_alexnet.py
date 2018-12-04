import tensorflow as tf

alexnet_url = None

def Alexnet(tensor):
    x = tf.keras.layers.ZeroPadding2D(2)(tensor)
    x = tf.keras.layers.Conv2D(64, 11, 4, activation='relu')(x)
    x = tf.keras.layers.MaxPool2D(3, 2)(x)
    x = tf.keras.layers.ZeroPadding2D(2)(x)
    x = tf.keras.layers.Conv2D(192, 5, activation='relu')(x)
    x = tf.keras.layers.MaxPool2D(3, 2)(x)
    x = tf.keras.layers.ZeroPadding2D(1)(x)
    x = tf.keras.layers.Conv2D(384, 3, activation='relu')(x)
    x = tf.keras.layers.ZeroPadding2D(1)(x)
    x = tf.keras.layers.Conv2D(256, 3, activation='relu')(x)
    x = tf.keras.layers.ZeroPadding2D(1)(x)
    x = tf.keras.layers.Conv2D(256, 3, activation='relu')(x)
    x = tf.keras.layers.MaxPool2D(3, 2)(x)
    return x

def get_alexnet(input_shape, include_top, pretrain_file, classes):
    if isinstance(pretrain_file, str) and include_top and classes != 1000:
        raise ValueError('If using `pretrain weights` with `include_top` as true, `classes` should be 1000')
    image = tf.keras.Input(shape=input_shape)
    x = Alexnet(image)
    if include_top:
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(4096, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(4096, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(classes)(x)
    model = tf.keras.Model(image, x, name='alexnet')
    if isinstance(pretrain_file, str):
        if tf.gfile.Exists(pretrain_file):
            model.load_weights(pretrain_file, by_name=True)
        else:
            tf.gfile.MakeDirs(pretrain_file)
            tf.gfile.DeleteRecursively(pretrain_file)
            tf.keras.utils.get_file(pretrain_file, alexnet_url)
            model.load_weights(pretrain_file, by_name=True)
    return model

def alexnet(input_shape, include_top=True, pretrain_file=False, classes=1000):
    return get_alexnet(input_shape, include_top, pretrain_file, classes)
