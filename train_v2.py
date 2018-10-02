# Trung-Hieu Tran @ IPVS
# 181002
# loss: mae val: mse

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
from utils.mod_epinet import MOD_EPINET


def set_seed(config):
    seed = config["seed"]
    set_random_seed(seed)
    np.random.seed(seed)
    # if not config["no_cuda"]:
    #     torch.cuda.manual_seed(seed)
    random.seed(seed)

def train(config):
    str_time = strftime("%y%m%d_%H%M%S",gmtime())
    log_dir = os.path.join(config['prefix_dir'],str_time)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    train_set, dev_set, test_set = Dataset_Generator.splits(config)
    print(" Summary: Train set %d , Dev set %d, Test set %d"%(len(train_set),
                                                              len(dev_set),
                                                              len(test_set)))
    model_lr= config['learning_rate']
    epi_model = MOD_EPINET(config)

    epi_model.summary()
    epi_model.save_model(os.path.join(log_dir,'model.json'))
    epi_model.export_plot(os.path.join(log_dir,'model.pdf'))

    # sgd = optimizers.SGD(lr=config['lr'][0],decay=config['weight_decay'],
    #                     nesterov=config['use_nesterov'], momentum=config['momentum'])
    optimizer = optimizers.RMSprop(lr=model_lr)
    epi_model.model.compile(optimizer=optimizer,
                      loss='mean_absolute_error',
                      metrics=['mse'])

    configPro = tf.ConfigProto(device_count={'GPU':2,'CPU':2})
    sess = tf.Session(config=configPro)
    keras.backend.set_session(sess)
    # tensorboard logs
    tensorboard = TensorBoard(log_dir=log_dir)
    # checkpoint
    filepath="%s/improvement-{epoch:02d}-{val_loss:.2f}.hdf5"%(log_dir)
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss',
                                 verbose=1, save_best_only=True,
                                 mode='auto')

    callbacks_list = [tensorboard,checkpoint]

    # keras.backend.get_session().run(tf.initialize_all_variables())
    keras.backend.get_session().run(tf.global_variables_initializer())
    if 'weight_file' in config:
        epi_model.load_weights(config['weight_file'])

    epi_model.model.fit_generator(generator=train_set,
                            steps_per_epoch=(len(train_set)),
                            epochs = config['n_epochs'],
                            verbose=1,
                            validation_data = dev_set,
                            validation_steps= (len(dev_set)),
                            use_multiprocessing=False,
                            workers=1,
                            max_queue_size=10,
                            callbacks=callbacks_list)
    epi_model.save_weights(os.path.join(log_dir,config['output_file']))

def main():
    config = {}
    # config['data_dir'] ='/home/trantu/lightfield/datasets/hci/full/additional'
    # config['disparity_dir'] = '/home/trantu/lightfield/datasets/hci/full/depths/additional'

    # config['data_dir'] ='/home/trantu/lightfield/local/hci/full/additional'
    # config['disparity_dir'] = '/home/trantu/lightfield/local/hci/full/depths/additional'
    config['patch_size'] = 29
    # config['stride'] = 6
    config['input_file'] = '/home/trantu/maps/pool1/data/epinet/train_5v_29p_17s.h5'
    # config['input_file'] = '/home/trantu/tmp/train_9v_23p_6s.h5'
    config['output_file'] = './weights.h5'
    # config['output_file'] = '/home/trantu/tmp/t9.h5'
    config['view_size'] = 5

    config['filt_num'] = 70
    config['conv_depth'] = 7

    # config['aug_shift'] = True
    # config['thres_patch'] = 0.03*255 # threshold to select good patch
    config['batch_size'] = 16
    config['dset_input'] = 'inputs'
    config['dset_label'] = 'labels'
    config['seed'] = 1234
    config['mode'] = 'train'
    config['n_epochs'] = 500
    config['learning_rate'] = 0.1**5

    config['prefix_dir'] = './logs/view_5_v2'

    if not os.path.exists(config['prefix_dir']):
        os.makedirs(config['prefix_dir'])

    # GPU setting ( gtx 1080ti - gpu0 )
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
    set_seed(config)
    if config["mode"] == "train":
        train(config)
    elif config["mode"] == "eval":
        print("TODO")

if __name__ == "__main__":
    main()
