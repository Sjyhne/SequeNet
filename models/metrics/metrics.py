import tensorflow as tf


def mean_iou(y_true, y_pred):
    y_true = tf.cast(y_true, tf.dtypes.float64)
    y_pred = tf.cast(y_pred, tf.dtypes.float64)
    i = tf.reduce_sum(y_pred * y_true, axis=(1, 2))
    u = tf.reduce_sum(y_pred + y_true, axis=(1, 2)) - i

    return tf.reduce_mean(i / u)


def dice_coefficient(y_true, y_pred, smooth=1):
    i = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    u = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3])
    dice = tf.reduce_mean((2. * i + smooth) / (u + smooth), axis=0)
    
    return dice