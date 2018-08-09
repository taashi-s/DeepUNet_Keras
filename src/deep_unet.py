from keras.models import Model
from keras.layers import Input, Conv2D, Activation, BatchNormalization
from keras.optimizers import Adam, SGD
from keras.utils import multi_gpu_model, plot_model
import keras.backend as KB

from layers import DownBlock, UpBlock
from dice_coefficient import DiceLossByClass# dice_coef_loss


class DeepUNet(object):
    def __init__(self, input_shape, internal_filter=32, depth=7, class_num=1):
        self.__input_shape = input_shape
        self.__class_num = class_num

        inputs = Input(self.__input_shape)
        conv_layer = Conv2D(internal_filter, 3, padding='same')(inputs)

        puls_layers = []
        encode_layer = conv_layer
        for d in range(depth):
            block_outputs = DownBlock(internal_filter=internal_filter*2
                                      , with_batch_norm=True
                                      #, dropout_rate=0.2
                                     )(encode_layer)
            encode_layer = block_outputs[0]
            puls_layers.append(block_outputs[1])

        decode_layer = encode_layer
        for puls_layer in reversed(puls_layers):
            decode_layer = UpBlock(internal_filter=internal_filter*2
                                   , with_batch_norm=True
                                   #, dropout_rate=0.5
                                  )([decode_layer, puls_layer])

        decode_layer = BatchNormalization()(decode_layer)
        decode_layer = Activation('relu')(decode_layer)
        #if class_num == 1:
        #    outputs = Conv2D(class_num, 1, activation='sigmoid', padding='same')(decode_layer)
        #else:
        #    outputs = Conv2D(class_num, 1, activation='softmax', padding='same')(decode_layer)
        outputs = Conv2D(class_num, 1, activation='sigmoid', padding='same')(decode_layer)
        #outputs = Conv2D(class_num, 1, activation='softmax', padding='same')(decode_layer)

        self.__model = Model(inputs=[inputs], outputs=[outputs])


    def comple_model(self):
        self.__model.compile(optimizer=Adam(lr=0.01), loss=DiceLossByClass(self.__input_shape, self.__class_num).dice_coef_loss)
        #self.__model.compile(optimizer=SGD(lr=0.01, momentum=0.99), loss=DiceLossByClass(self.__input_shape, self.__class_num).dice_coef_loss)


    def get_model(self, with_comple=False):
        if with_comple:
            self.comple_model()
        return self.__model


    def get_parallel_model(self, gpu_num, with_comple=False):
        self.__model = multi_gpu_model(self.__model, gpus=gpu_num)
        return self.get_model(with_comple)


    def show_model_summary(self):
        self.__model.summary()


    def plot_model_summary(self, file_name):
        plot_model(self.__model, to_file=file_name)
