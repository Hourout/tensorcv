import tensorflow as tf

alexnet_url = None

def alexnet(input_shape, include_top=True, pretrain_file=False, classes=1000):
    if isinstance(pretrain_file, str) and include_top and classes != 1000:
        raise ValueError('If using `pretrain weights` with `include_top` as true, `classes` should be 1000')
    image = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(64, 11, 4, 'same', activation='relu', name='conv1')(image)
    x = tf.keras.layers.MaxPool2D(3, 2, name='pool1')(x)
    x = tf.keras.layers.Conv2D(192, 5, 1, 'same', activation='relu', name='conv2')(x)
    x = tf.keras.layers.MaxPool2D(3, 2, name='pool2')(x)
    x = tf.keras.layers.Conv2D(384, 3, 1, 'same', activation='relu', name='conv3')(x)
    x = tf.keras.layers.Conv2D(256, 3, 1, 'same', activation='relu', name='conv4')(x)
    x = tf.keras.layers.Conv2D(256, 3, 1, 'same', activation='relu', name='conv5')(x)
    if include_top:
        x = tf.keras.layers.MaxPool2D(3, 2, name='pool3')(x)
        x = tf.keras.layers.Flatten(name='flatten')(x)
        x = tf.keras.layers.Dense(4096, activation='relu', name='liner1')(x)
        x = tf.keras.layers.Dropout(0.5, name='drop1')(x)
        x = tf.keras.layers.Dense(4096, activation='relu', name='liner2')(x)
        x = tf.keras.layers.Dropout(0.5, name='drop1')(x)
        x = tf.keras.layers.Dense(classes, name='predictions')(x)
    model = tf.keras.Model(image, x, name='alexnet')
    if isinstance(pretrain_file, str):
        if tf.gfile.Exists(pretrain_file):
            model.load_weights(pretrain_file)
        else:
            tf.gfile.MakeDirs(pretrain_file)
            tf.gfile.DeleteRecursively(pretrain_file)
            tf.keras.utils.get_file(pretrain_file, alexnet_url)
            model.load_weights(pretrain_file)
    return model
