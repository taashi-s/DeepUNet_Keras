from keras.models import Model
from keras.engine.topology import Input
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam

from layers import DownBlock, UpBlock
from dice_coefficient import dice_coef_loss


class DeepUNet(object):
    def __init__(self, input_shape, internal_filter=32, depth=7):
        self.__input_shape = input_shape

        inputs = Input(self.__input_shape)
        conv_layer = Conv2D(internal_filter, 3, padding='same')(inputs)

        puls_layers = []
        encode_layer = conv_layer
        for d in range(depth):
            encode_layer, puls_layer = DownBlock()(encode_layer)
            puls_layers.append(puls_layer)

        decode_layer = encode_layer
        for d in range(depth):
            decode_layer = UpBlock()([decode_layer, puls_layers[depth - 1 - d]])

        outputs = Conv2D(1, 1, activation='softmax', padding='same')(decode_layer)

        self.__model = Model(inputs=[inputs], outputs=[outputs])


    def comple_model(self):
        self.__model.compile(optimizer=Adam(), loss=dice_coef_loss)        


    def get_model(self, with_comple=False):
        if with_comple:
            self.comple_model()
        return self.__model
