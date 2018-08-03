import keras.backend as KB
import keras.engine.base_layer as KELayer
from keras.layers import Conv2D, Activation, Dropout, BatchNormalization, Add, MaxPooling2D


class DownBlock(KELayer.Layer):
    def __init__(self, **kwargs):
        super(DownBlock, self).__init__(**kwargs)


    def call(self, inputs, **kwargs):
        return self.__down_block(inputs)


    def __down_block(self, input_layer):
        filters = input_layer.get_shape().as_list()[3]
        layer_1 = self.__conv_block(64, input_layer)
        layer_2 = self.__conv_block(filters, layer_1)
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


    def compute_output_shape(self, input_shape):
        return [ (None, input_shape[1] // 2, input_shape[2] // 2, input_shape[3])
               , (None, input_shape[1], input_shape[2], input_shape[3])
               ]
