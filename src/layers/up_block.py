import keras.backend as KB
import keras.engine.base_layer as KELayer
from keras.layers import Conv2D, Activation, Dropout, BatchNormalization, Concatenate, Add, UpSampling2D 


class UpBlock(KELayer.Layer):
    def __init__(self, **kwargs):
        super(UpBlock, self).__init__(**kwargs)


    def call(self, inputs, **kwargs):
        return self.__up_block(*inputs)


    def __up_block(self, input_layer, concat_layer):
        filters = input_layer.get_shape().as_list()[3]

        upsample = UpSampling2D(2)(input_layer)
        layer = Activation('relu')(upsample)
        layer = Concatenate()([layer, concat_layer])
        layer = Conv2D(64, 3, strides=1, padding='same')(layer)

        layer = Activation('relu')(layer)
        layer = Conv2D(filters, 3, strides=1, padding='same')(layer)

        puls = Add()([layer, upsample])
        return puls 


    def compute_output_shape(self, input_shape):
        return [(None, input_shape[0][1] * 2, input_shape[0][2] * 2, input_shape[0][3])]
