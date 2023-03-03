import tensorflow as tf
from keras.layers import Add, Conv2D, Input, Lambda, Rescaling
from keras.models import Model


class EDSR_model_builder(Model):
    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, y_pred)

        return {m.name: m.result() for m in self.metrics}

    def predict_step(self, data):
        x = tf.cast(tf.expand_dims(data, axis=0), tf.float32)
        pred = self(x, training=False)
        pred = tf.clip_by_value(pred, 0, 255)
        pred = tf.round(pred)
        pred = tf.cast(tf.squeeze(pred, axis=0), tf.uint8)

        return pred

def res_block(Input_layer, filter=64):
    res_x = Conv2D(filters=filter, kernel_size=3, activation='relu', padding='same')(Input_layer)
    res_x = Conv2D(filters=filter, kernel_size=3, padding='same')(res_x)
    res_x = Lambda(function=lambda x: x * 0.1)(res_x)
    result = Add()([Input_layer, res_x])

    return result

def upsampling(Input_layer, filter=64, scale=2, **sky):
    up_x = Conv2D(filter * (scale ** 2), kernel_size=3, padding='same', **sky)(Input_layer)
    up_x = tf.nn.depth_to_space(up_x, block_size=scale)
    up_x = Conv2D(filter * (scale ** 2), kernel_size=3, padding='same', **sky)(up_x)
    up_x = tf.nn.depth_to_space(up_x, block_size=scale)

    return up_x

def build_edsr(filter=64, channel=3, num_res=16):
    input = Input((None, None, channel))
    x = Rescaling(scale=1.0/255.0)(input)
    x = res_x = Conv2D(filter, kernel_size=3, padding='same')(x)

    for _ in range(num_res):
        res_x = res_block(res_x)

    res_x = Conv2D(filter, kernel_size=3, padding='same')(res_x)
    x = Add()([x, res_x])

    up_x = upsampling(x)
    output = Conv2D(channel, kernel_size=3, padding='same')(up_x)
    output = Rescaling(255)(output)

    model = EDSR_model_builder(input, output)

    return model
