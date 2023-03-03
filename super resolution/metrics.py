import tensorflow as tf

def PSNR(y_true, y_pred):
    return tf.image.psnr(y_true, y_pred, max_val=255)[0]

def SSIM(y_true, y_pred):
    return tf.image.ssim(y_true, y_pred, max_val=255)[0]