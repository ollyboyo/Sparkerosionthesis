import tensorflow as tf
import tensorflow.keras.backend as kb


def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = kb.flatten(y_true)
    y_pred_f = kb.flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    loss = 1.0 - dice_coef(y_true, y_pred)
    return loss
