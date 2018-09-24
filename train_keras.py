# Trung-Hieu Tran @ IPVS
# 180921

# from utils.epinet_dataset import  EPI_Dataset
from __future__ import print_function
from chainmap import ChainMap
import argparse
import os
import random
import sys

import numpy as np
from keras import optimizers

# reproduceable with random
from numpy.random import seed
from tensorflow import set_random_seed
import tensorflow as tf
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
import keras
from utils.epinet_dataset import Dataset_Generator
from time import strftime, gmtime
from utils.keras_epinet import define_epinet_v1


def set_seed(config):
    seed = config["seed"]
    set_random_seed(seed)
    np.random.seed(seed)
    # if not config["no_cuda"]:
    #     torch.cuda.manual_seed(seed)
    random.seed(seed)

def train(config):
    train_set, dev_set, test_set = Dataset_Generator.splits(config)
    print(" Summary: Train set %d , Dev set %d, Test set %d"%(len(train_set),
                                                              len(dev_set),
                                                              len(test_set)))
    image_h = config['patch_size']
    image_w = config['patch_size']
    view_size = config['view_size']
    model_conv_depth=7
    model_filt_num=70
    model_learning_rate=0.1**5
    model_512=define_epinet_v1((image_h, image_w),
                            view_size,
                            model_conv_depth,
                            model_filt_num,
                            model_learning_rate)

    model_512.summary()
    # sgd = optimizers.SGD(lr=config['lr'][0],decay=config['weight_decay'],
    #                     nesterov=config['use_nesterov'], momentum=config['momentum'])
    optimizer = optimizers.RMSprop(lr=model_learning_rate)
    model_512.compile(optimizer=optimizer,
                      loss='mean_absolute_error',
                      metrics=['mae'])

    configPro = tf.ConfigProto(device_count={'GPU':2,'CPU':2})
    sess = tf.Session(config=configPro)
    keras.backend.set_session(sess)
    # tensorboard logs
    str_time = strftime("%y%m%d_%H%M%S",gmtime())
    tensorboard = TensorBoard(log_dir="logs/v1/v1_{}".format(str_time))
    callbacks_list = [tensorboard]

    # keras.backend.get_session().run(tf.initialize_all_variables())
    keras.backend.get_session().run(tf.global_variables_initializer())
    if 'weight_file' in config:
        model_512.load_weights(config['weight_file'])

    model_512.fit_generator(generator=train_set,
                            steps_per_epoch=(len(train_set)),
                            epochs = config['n_epochs'],
                            verbose=1,
                            validation_data = dev_set,
                            validation_steps= (len(dev_set)),
                            use_multiprocessing=False,
                            workers=1,
                            max_queue_size=10,
                            callbacks=callbacks_list)
    model_512.save(config['output_file'])



def main():
    config = {}
    # config['data_dir'] ='/home/trantu/lightfield/datasets/hci/full/additional'
    # config['disparity_dir'] = '/home/trantu/lightfield/datasets/hci/full/depths/additional'

    # config['data_dir'] ='/home/trantu/lightfield/local/hci/full/additional'
    # config['disparity_dir'] = '/home/trantu/lightfield/local/hci/full/depths/additional'
    config['patch_size'] = 61
    # config['stride'] = 6
    #config['input_file'] = '/home/trantu/maps/pool1/data/epinet/train_5v_23p_6s.h5'
    config['input_file'] = '/home/trantu/tmp/train_9v_23p_6s.h5'
    config['output_file'] = './weights.h5'
    # config['output_file'] = '/home/trantu/tmp/t9.h5'
    config['view_size'] = 9

    # config['aug_shift'] = True
    # config['thres_patch'] = 0.03*255 # threshold to select good patch
    config['batch_size'] = 16
    config['dset_input'] = 'inputs'
    config['dset_label'] = 'labels'
    config['seed'] = 1234
    config['mode'] = 'train'
    config['n_epochs'] = 100

    set_seed(config)
    if config["mode"] == "train":
        train(config)
    elif config["mode"] == "eval":
        print("TODO")

if __name__ == "__main__":
    main()
