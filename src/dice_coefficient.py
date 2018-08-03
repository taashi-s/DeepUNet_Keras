import keras.backend as KB
import tensorflow as tf

def dice_coef(y_true, y_pred):
    y_true = KB.flatten(y_true)
    y_pred = KB.flatten(y_pred)
    intersection = KB.sum(y_true * y_pred)
    return (2.0 * intersection + 1) / (KB.sum(y_true) + KB.sum(y_pred) + 1)


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred) 
