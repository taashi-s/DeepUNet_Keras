import os
import numpy as np
from matplotlib import pyplot
import keras.callbacks as KC
import math

from deep_unet import DeepUNet
from images_loader import load_images, save_images
from option_parser import get_option
from data_generator import DataGenerator
from history_checkpoint_callback import HistoryCheckpoint


CLASS_NUM = 3
DEPTH = 5
PADDING = 1
INPUT_IMAGE_SHAPE = (256 + (PADDING * 2), 256 + (PADDING * 2), 3)
BATCH_SIZE = 20
EPOCHS = 1000
GPU_NUM = 4

INTERNAL_FILTER = 64 

DIR_MODEL = os.path.join('..', 'model')
DIR_INPUTS = os.path.join('..', 'inputs')
DIR_OUTPUTS = os.path.join('..', 'outputs')
#DIR_TEACHERS = os.path.join('..', 'teachers_gray')
DIR_TEACHERS = os.path.join('..', 'teachers')
DIR_PREDICTS = os.path.join('..', 'predict_data')

File_MODEL = 'segmentation_model.hdf5'


def train(gpu_num=None):
    #print('input data loading ...', )
    #(_, inputs) = load_images(DIR_INPUTS, INPUT_IMAGE_SHAPE)
    #print('... loaded .', )
    #print('teacher data loading ...', )
    #(_, teachers) = load_images(DIR_TEACHERS, INPUT_IMAGE_SHAPE)
    #print('... loaded .', )

    network = DeepUNet(INPUT_IMAGE_SHAPE, internal_filter=INTERNAL_FILTER, depth=DEPTH, class_num=CLASS_NUM)
    model_filename = os.path.join(DIR_MODEL, File_MODEL)
    network.plot_model_summary('../model_plot.png')
    network.show_model_summary()
    if isinstance(gpu_num, int):
        model = network.get_parallel_model(gpu_num, with_comple=True)
    else:
        model = network.get_model(with_comple=True)

    callbacks = [ KC.TensorBoard()
                , HistoryCheckpoint(filepath='LearningCurve_{history}.png'
                                    , verbose=1
                                    , period=10
                                   )
                , KC.ModelCheckpoint(filepath=model_filename
                                     , verbose=1
                                     , save_weights_only=True
                                     #, save_best_only=True
                                     , period=10
                                    )
                ]

    print('data generateing ...')
    train_generator = DataGenerator(DIR_INPUTS, DIR_TEACHERS, INPUT_IMAGE_SHAPE, include_padding=(PADDING, PADDING))
    inputs, teachers = train_generator.generate_data()
    print('... generated')

    history = model.fit(inputs, teachers, batch_size=BATCH_SIZE, epochs=EPOCHS
                        , shuffle=True, verbose=1, callbacks=callbacks)
    model.save_weights(model_filename)
    plotLearningCurve(history)


def train_with_generator(gpu_num=None):
    network = DeepUNet(INPUT_IMAGE_SHAPE, internal_filter=INTERNAL_FILTER, depth=DEPTH, class_num=CLASS_NUM)
    if isinstance(gpu_num, int):
        model = network.get_parallel_model(gpu_num, with_comple=True)
    else:
        model = network.get_model(with_comple=True)
    model.summary()
    model_filename = os.path.join(DIR_MODEL, File_MODEL)
    callbacks = [ KC.TensorBoard()
                , HistoryCheckpoint(filepath='LearningCurve_{history}.png'
                                    , verbose=1
                                    , period=10
                                   )
                , KC.ModelCheckpoint(filepath=model_filename
                                     , verbose=1
                                     , save_weights_only=True
                                     #, save_best_only=True
                                     , period=10
                                    )
                ]

    train_generator = DataGenerator(DIR_INPUTS, DIR_TEACHERS, INPUT_IMAGE_SHAPE)
    train_data_num = train_generator.data_size()


    print('fix ...')
    his = model.fit_generator(train_generator.generator(batch_size=BATCH_SIZE)
                              , steps_per_epoch=math.ceil(train_data_num / BATCH_SIZE)
                              , epochs=EPOCHS
                              , verbose=1
                              , use_multiprocessing=True
                              , callbacks=callbacks
                              #, validation_data=valid_generator
                              #, validation_steps=math.ceil(valid_data_num / BATCH_SIZE)
                             )
    print('model saveing ...')
    model.save_weights(model_filename)
    print('... saved')
    plotLearningCurve(his)


def plotLearningCurve(history):
    """ saveLearningCurve """
    x = range(EPOCHS)
    pyplot.plot(x, history.history['loss'], label="loss")
    pyplot.title("loss")
    pyplot.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    lc_name = 'LearningCurve'
    pyplot.savefig(lc_name + '.png')
    pyplot.close()


def predict(input_dir, gpu_num=None):
    h, w, c = INPUT_IMAGE_SHAPE
    org_h, org_w = h - (PADDING * 2), w - (PADDING * 2)
    (file_names, inputs) = load_images(input_dir, (org_h, org_w, c))
    inputs = np.pad(inputs, [(0, 0), (PADDING, PADDING), (PADDING, PADDING), (0, 0)], 'constant', constant_values=0)

    network = DeepUNet(INPUT_IMAGE_SHAPE, internal_filter=INTERNAL_FILTER, depth=DEPTH, class_num=CLASS_NUM)
    if isinstance(gpu_num, int):
        model = network.get_parallel_model(gpu_num)
    else:
        model = network.get_model()
#    model.summary()
    print('loading weghts ...')
    model.load_weights(os.path.join(DIR_MODEL, File_MODEL))
    print('... loaded')
    print('predicting ...')
    preds = model.predict(inputs, BATCH_SIZE)
    print('... predicted')

    print('result saveing ...')
    preds = preds[:, PADDING:org_h, PADDING:org_w, :]

    save_images(DIR_OUTPUTS, preds, file_names)
    print('... finish .')


if __name__ == '__main__':
    args = get_option(EPOCHS)
    EPOCHS = args.epoch

    if not(os.path.exists(DIR_MODEL)):
        os.mkdir(DIR_MODEL)
    if not(os.path.exists(DIR_OUTPUTS)):
        os.mkdir(DIR_OUTPUTS)

    train(gpu_num=GPU_NUM)
    #train_with_generator(gpu_num=GPU_NUM)

    #predict(DIR_INPUTS, gpu_num=GPU_NUM)
    #predict(DIR_PREDICTS, gpu_num=GPU_NUM)
