import math
import keras.backend as KB
import keras.engine.base_layer as KELayer
from keras.layers import Conv2D, Activation, Dropout, BatchNormalization, Concatenate, Add, UpSampling2D, Cropping2D, ZeroPadding2D


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
        upsample, concat_layer = self.__adjustment_shape(upsample, concat_layer)
        layer = upsample
        if self.__with_batch_norm:
            layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)
        #if isinstance(self.__dropout_rate, float):
        #    layer = Dropout(self.__dropout_rate)(layer)
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


    def __adjustment_shape(self, input_layer, concat_layer):
        _, h_i, w_i, _ = input_layer.get_shape().as_list()
        _, h_c, w_c, _ = concat_layer.get_shape().as_list()

        if h_i < h_c:
            #crop_hs = self.__get_crop_size(h_i, h_c)
            #concat_layer = Cropping2D(cropping=(crop_hs, (0, 0)))(concat_layer)
            pad_hs = self.__get_crop_size(h_i, h_c)
            input_layer = ZeroPadding2D(padding=(pad_hs, (0, 0)))(input_layer)
        elif h_c < h_i:
            #crop_hs = self.__get_crop_size(h_i, h_c)
            #input_layer = Cropping2D(cropping=(crop_hs, (0, 0)))(input_layer)
            pad_hs = self.__get_crop_size(h_i, h_c)
            concat_layer = ZeroPadding2D(padding=(pad_hs, (0, 0)))(concat_layer)

        if w_i < w_c:
            #crop_ws = self.__get_crop_size(w_i, w_c)
            #concat_layer = Cropping2D(cropping=((0, 0), crop_ws))(concat_layer)
            pad_ws = self.__get_crop_size(w_i, w_c)
            input_layer = ZeroPadding2D(padding=((0, 0), pad_ws))(input_layer)
        elif w_c < w_i:
            #crop_ws = self.__get_crop_size(w_i, w_c)
            #input_layer = Cropping2D(cropping=((0, 0), crop_ws))(input_layer)
            pad_ws = self.__get_crop_size(w_i, w_c)
            concat_layer = ZeroPadding2D(padding=((0, 0), pad_ws))(concat_layer)

        return input_layer, concat_layer


    def __get_crop_size(self, target, crop):
        pad = (crop - target) / 2
        crop_size = (int(pad), int(pad))
        if pad != int(pad):
            crop_size = (math.ceil(pad), int(pad))
        return crop_size
