import tensorflow as tf

alexnet_url = None

def alexnet(pretrain_file=False, input_shape=None, classes=1000):
    image = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(64, 11, 4, 'same', activation='relu')(image)
    x = tf.keras.layers.MaxPool2D(3, 2)(x)
    x = tf.keras.layers.Conv2D(192, 5, 1, 'same', activation='relu')(x)
    x = tf.keras.layers.MaxPool2D(3, 2)(x)
    x = tf.keras.layers.Conv2D(384, 3, 1, 'same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(256, 3, 1, 'same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(256, 3, 1, 'same', activation='relu')(x)
    x = tf.keras.layers.MaxPool2D(3, 2)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(4096, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(4096, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(classes)(x)
    model = tf.keras.Model(image, x)
    if isinstance(pretrain_file, str):
        if tf.gfile.Exists(pretrain_file):
            model.load_weights(pretrain_file)
        else:
            tf.keras.utils.get_file(pretrain_file, alexnet_url)
            model.load_weights(pretrain_file)
    return model
