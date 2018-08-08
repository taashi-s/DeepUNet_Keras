import keras.backend as KB
import keras.engine.base_layer as KELayer
from keras.layers import Conv2D, Activation, Dropout, BatchNormalization, Concatenate, Add, UpSampling2D 


class UpBlock():
    def __init__(self, internal_filter=64, with_batch_norm=False, dropout_rate=None, **kwargs):
        self.__internal_filter = internal_filter
        self.__with_batch_norm = with_batch_norm
        self.__dropout_rate = dropout_rate


    def __call__(self, inputs):
        return self.__up_block(*inputs)


    def __up_block(self, input_layer, concat_layer):
        filters = input_layer.get_shape().as_list()[3]

        upsample = UpSampling2D()(input_layer)
        layer = upsample
        if self.__with_batch_norm:
            layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)
        if isinstance(self.__dropout_rate, float):
            layer = Dropout(self.__dropout_rate)(layer)
        layer = Concatenate()([layer, concat_layer])
        layer = Conv2D(self.__internal_filter, 3, strides=1, padding='same')(layer)

        if self.__with_batch_norm:
            layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)
        if isinstance(self.__dropout_rate, float):
            layer = Dropout(self.__dropout_rate)(layer)
        layer = Conv2D(filters, 3, strides=1, padding='same')(layer)

        puls = Add()([layer, upsample])
        return puls 
