import keras.backend as KB
import keras.engine.base_layer as KELayer
from keras.layers import Conv2D, Activation, Dropout, BatchNormalization, Add, MaxPooling2D


class DownBlock():
    def __init__(self, internal_filter=64, with_batch_norm=False, dropout_rate=None, **kwargs):
        self.__internal_filter = internal_filter
        self.__with_batch_norm = with_batch_norm
        self.__dropout_rate = dropout_rate


    def __call__(self, inputs):
        return self.__down_block(inputs)


    def __down_block(self, input_layer):
        filters = input_layer.get_shape().as_list()[3]
        layer_1 = self.__conv_block(self.__internal_filter, input_layer
                                    , with_batch_norm=self.__with_batch_norm
                                    , dropout_rate=self.__dropout_rate)
        layer_2 = self.__conv_block(filters, layer_1
                                    , with_batch_norm=False
                                    , dropout_rate=self.__dropout_rate)
        puls = Add()([layer_2, input_layer])
        pool = MaxPooling2D()(puls)
        return [pool, puls]


    def __conv_block(self, filters, input_layer, kernel_size=3, strides=1, with_batch_norm=False, dropout_rate=None):
        layer = input_layer
        if with_batch_norm:
            layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)
        if isinstance(dropout_rate, float):
            layer = Dropout(dropout_rate)(layer)
        layer = Conv2D(filters, kernel_size, strides=strides, padding='same')(layer)
        return layer
