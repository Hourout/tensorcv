import tensorflow as tf

__all__ = ['vgg11', 'vgg13', 'vgg16', 'vgg19', 'vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn']

vgg_url = {'vgg11': None,
           'vgg13': None,
           'vgg16': None,
           'vgg19': None,
           'vgg11_bn': None,
           'vgg13_bn': None,
           'vgg16_bn': None,
           'vgg19_bn': None}

vgg_spec = {11: ([1, 1, 2, 2, 2], [64, 128, 256, 512, 512]),
            13: ([2, 2, 2, 2, 2], [64, 128, 256, 512, 512]),
            16: ([2, 2, 3, 3, 3], [64, 128, 256, 512, 512]),
            19: ([2, 2, 4, 4, 4], [64, 128, 256, 512, 512])}

def VGG(num_layers, mode, input_shape, include_top, pretrain_file, classes, use_bn):
    if isinstance(pretrain_file, str) and include_top and classes != 1000:
        raise ValueError('If using `pretrain weights` with `include_top` as true, `classes` should be 1000')
    layers, filters = vgg_spec[num_layers]
    assert len(layers) == len(filters)
    image = tf.keras.Input(shape=input_shape)
    for i, num in enumerate(layers):
        for j in range(num):
            if (i==0)&(j==0):
                x = tf.keras.layers.Conv2D(filters[i], 3, 1, 'same', name='conv'+str(i+1)+'_'+str(j+1))(image)
            else:
                x = tf.keras.layers.Conv2D(filters[i], 3, 1, 'same', name='conv'+str(i+1)+'_'+str(j+1))(x)
            if use_bn:
                x = tf.keras.layers.BatchNormalization(name='bn'+str(i+1)+'_'+str(j+1))(x)
            x = tf.keras.layers.ReLU(name='relu'+str(i+1)+'_'+str(j+1))(x)    
        x = tf.keras.layers.MaxPool2D(2, 2, name='pool'+str(i+1)+'_'+str(j+1))(x)
    if include_top:
        x = tf.keras.layers.Flatten(name='flatten')(x)
        x = tf.keras.layers.Dense(4096, 'relu', name='liner1')(x)
        x = tf.keras.layers.Dropout(0.5, name='drop1')(x)
        x = tf.keras.layers.Dense(4096, 'relu', name='liner2')(x)
        x = tf.keras.layers.Dropout(0.5, name='drop2')(x)
        x = tf.keras.layers.Dense(classes, name='predictions')(x)
    model = tf.keras.Model(image, x, name=mode)
    if isinstance(pretrain_file, str):
        if tf.gfile.Exists(pretrain_file):
            model.load_weights(pretrain_file, by_name=True)
        else:
            tf.gfile.MakeDirs(pretrain_file)
            tf.gfile.DeleteRecursively(pretrain_file)
            tf.keras.utils.get_file(pretrain_file, vgg_url(mode))
            model.load_weights(pretrain_file, by_name=True)
    return model

def vgg11(input_shape, include_top=True, pretrain_file=False, classes=1000):
    r"""VGG-11 model from the `"Very Deep Convolutional Networks for Large-Scale Image Recognition"
    <https://arxiv.org/abs/1409.1556>`_ paper.
    
    Parameters
    ----------
    input_shape : tuple of 3 ints, input image shape.
    include_top : whether to include the fully-connected layer at the top of the network.
    pretrain_file : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
        String value must be absolute path of the pretrained weights file,
        if pretrain file is not exists, will auto create dir and download pretrained weights.
    classes: optional number of classes to classify images into, 
             only to be specified if `include_top` is True.
    """
    return VGG(11, 'vgg11', input_shape, include_top, pretrain_file, classes, False)

def vgg13(input_shape, include_top=True, pretrain_file=False, classes=1000):
    r"""VGG-13 model from the `"Very Deep Convolutional Networks for Large-Scale Image Recognition"
    <https://arxiv.org/abs/1409.1556>`_ paper.
    
    ----------
    input_shape : tuple of 3 ints, input image shape.
    include_top : whether to include the fully-connected layer at the top of the network.
    pretrain_file : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
        String value must be absolute path of the pretrained weights file,
        if pretrain file is not exists, will auto create dir and download pretrained weights.
    classes: optional number of classes to classify images into, 
             only to be specified if `include_top` is True.
    """
    return VGG(13, 'vgg13', input_shape, include_top, pretrain_file, classes, False)

def vgg16(input_shape, include_top=True, pretrain_file=False, classes=1000):
    r"""VGG-16 model from the `"Very Deep Convolutional Networks for Large-Scale Image Recognition"
    <https://arxiv.org/abs/1409.1556>`_ paper.
    
    ----------
    input_shape : tuple of 3 ints, input image shape.
    include_top : whether to include the fully-connected layer at the top of the network.
    pretrain_file : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
        String value must be absolute path of the pretrained weights file,
        if pretrain file is not exists, will auto create dir and download pretrained weights.
    classes: optional number of classes to classify images into, 
             only to be specified if `include_top` is True.
    """
    return VGG(16, 'vgg16', input_shape, include_top, pretrain_file, classes, False)

def vgg19(input_shape, include_top=True, pretrain_file=False, classes=1000):
    r"""VGG-19 model from the `"Very Deep Convolutional Networks for Large-Scale Image Recognition"
    <https://arxiv.org/abs/1409.1556>`_ paper.
    
    ----------
    input_shape : tuple of 3 ints, input image shape.
    include_top : whether to include the fully-connected layer at the top of the network.
    pretrain_file : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
        String value must be absolute path of the pretrained weights file,
        if pretrain file is not exists, will auto create dir and download pretrained weights.
    classes: optional number of classes to classify images into, 
             only to be specified if `include_top` is True.
    """
    return VGG(19, 'vgg19', input_shape, include_top, pretrain_file, classes, False)

def vgg11_bn(input_shape, include_top=True, pretrain_file=False, classes=1000):
    r"""VGG-11 model with batch normalization from the
    `"Very Deep Convolutional Networks for Large-Scale Image Recognition"
    <https://arxiv.org/abs/1409.1556>`_ paper.
    
    ----------
    input_shape : tuple of 3 ints, input image shape.
    include_top : whether to include the fully-connected layer at the top of the network.
    pretrain_file : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
        String value must be absolute path of the pretrained weights file,
        if pretrain file is not exists, will auto create dir and download pretrained weights.
    classes: optional number of classes to classify images into, 
             only to be specified if `include_top` is True.
    """
    return VGG(11, 'vgg11_bn', input_shape, include_top, pretrain_file, classes, True)

def vgg13_bn(input_shape, include_top=True, pretrain_file=False, classes=1000):
    r"""VGG-13 model with batch normalization from the
    `"Very Deep Convolutional Networks for Large-Scale Image Recognition"
    <https://arxiv.org/abs/1409.1556>`_ paper.
    
    ----------
    input_shape : tuple of 3 ints, input image shape.
    include_top : whether to include the fully-connected layer at the top of the network.
    pretrain_file : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
        String value must be absolute path of the pretrained weights file,
        if pretrain file is not exists, will auto create dir and download pretrained weights.
    classes: optional number of classes to classify images into, 
             only to be specified if `include_top` is True.
    """
    return VGG(13, 'vgg13_bn', input_shape, include_top, pretrain_file, classes, True)

def vgg16_bn(input_shape, include_top=True, pretrain_file=False, classes=1000):
    r"""VGG-16 model with batch normalization from the
    `"Very Deep Convolutional Networks for Large-Scale Image Recognition"
    <https://arxiv.org/abs/1409.1556>`_ paper.
    
    ----------
    input_shape : tuple of 3 ints, input image shape.
    include_top : whether to include the fully-connected layer at the top of the network.
    pretrain_file : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
        String value must be absolute path of the pretrained weights file,
        if pretrain file is not exists, will auto create dir and download pretrained weights.
    classes: optional number of classes to classify images into, 
             only to be specified if `include_top` is True.
    """
    return VGG(16, 'vgg16_bn', input_shape, include_top, pretrain_file, classes, True)

def vgg19_bn(input_shape, include_top=True, pretrain_file=False, classes=1000):
    r"""VGG-19 model with batch normalization from the
    `"Very Deep Convolutional Networks for Large-Scale Image Recognition"
    <https://arxiv.org/abs/1409.1556>`_ paper.
    
    ----------
    input_shape : tuple of 3 ints, input image shape.
    include_top : whether to include the fully-connected layer at the top of the network.
    pretrain_file : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
        String value must be absolute path of the pretrained weights file,
        if pretrain file is not exists, will auto create dir and download pretrained weights.
    classes: optional number of classes to classify images into, 
             only to be specified if `include_top` is True.
    """
    return VGG(19, 'vgg19_bn', input_shape, include_top, pretrain_file, classes, True)
