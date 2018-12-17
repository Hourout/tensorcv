import tensorflow as tf

def BBoxCornerToCenter(tensor, axis=-1, split=False):
    xmin, ymin, xmax, ymax = tf.split(tensor, 4, axis)
    width = tf.math.subtract(xmax, xmin)
    height = tf.math.subtract(ymax, ymin)
    x = tf.math.add(xmin, tf.math.divide(width, 2))
    y = tf.math.add(ymin, tf.math.divide(height, 2))
    if not split:
        return tf.concat([x, y, width, height], axis)
    else:
        return x, y, width, height
    
def BBoxCenterToCorner(tensor, axis=-1, split=False):
    x, y, w, h = tf.split(tensor, 4, axis)
    hw = tf.math.divide(w, 2)
    hh = tf.math.divide(h, 2)
    xmin = tf.math.subtract(x, hw)
    ymin = tf.math.subtract(y, hh)
    xmax = tf.math.add(x, hw)
    ymax = tf.math.add(y, hh)
    if not split:
        return tf.concat([xmin, ymin, xmax, ymax], axis)
    else:
        return xmin, ymin, xmax, ymax
    
def BBoxSplit(tensor, axis=-1):
    return tf.split(tensor, 4, axis)

def BBoxArea(tensor, axis=-1, fmt='corner'):
    if fmt.lower() == 'corner':
        _, _, width, height = BBoxCornerToCenter(tensor, split=True)
    elif fmt.lower() == 'center':
        _, _, width, height = BBoxSplit(tensor, axis=axis)
    else:
        raise ValueError("Unsupported format: {}. Use 'corner' or 'center'.".format(fmt))
    width = tf.where(tf.math.greater(width, 0), width, tf.zeros_like(width))
    height = tf.where(tf.math.greater(height, 0), height, tf.zeros_like(height))
    return tf.math.multiply(width, height)

def BBoxBatchIOU(tensor1, tensor2, axis=-1, fmt='corner', offset=0, eps=1e-15):
    if fmt.lower() == 'center':
        al, at, ar, ab = BBoxCenterToCorner(tensor1, split=True)
        bl, bt, br, bb = BBoxCenterToCorner(tensor2, split=True)
    elif fmt.lower() == 'corner':
        al, at, ar, ab = BBoxSplit(tensor1, axis=axis, squeeze_axis=True)
        bl, bt, br, bb = BBoxSplit(tensor2, axis=axis, squeeze_axis=True)
    else:
        raise ValueError("Unsupported format: {}. Use 'corner' or 'center'.".format(fmt))
    
    left = tf.math.maximum(tf.expand_dims(al, -1), tf.expand_dims(bl, -2))
    right = tf.math.minimum(tf.expand_dims(ar, -1), tf.expand_dims(br, -2))
    top = tf.math.maximum(tf.expand_dims(at, -1), tf.expand_dims(bt, -2))
    bot = tf.math.minimum(tf.expand_dims(ab, -1), tf.expand_dims(bb, -2))

    iw = tf.clip_by_value(tf.math.add(tf.math.subtract(right, left), offset), 0, 6.55040e+04)
    ih = tf.clip_by_value(tf.math.add(tf.math.subtract(bot, top), offset), 0, 6.55040e+04)
    i = tf.math.multiply(iw, ih)

    area_a = tf.expand_dims(tf.math.multiply(tf.math.add(tf.math.subtract(ar, al), offset), tf.math.add(tf.math.subtract(ab, at), offset)), -1)
    area_b = tf.expand_dims(tf.math.multiply(tf.math.add(tf.math.subtract(br, bl), offset), tf.math.add(tf.math.subtract(bb, bt), offset)), -2)
    union = tf.math.subtract(tf.math.add(area_a, area_b) - i)
    return tf.math.divide(i, tf.math.add(union, eps))

def BBoxClipToImage():
    x = tf.math.maximum(x, 0.0)
    window = tf.expand_dims(tf.slice(tf.shape(img), begin=2), 0)
    m = tf.manip.reshape(tf.tile(tf.manip.reverse(window, axis=1), [2]), [0, -4, 1, -1])
    return tf.math.minimum(x, tf.cast(m, dtype=tf.float32))
