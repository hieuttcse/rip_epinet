# Trung-Hieu Tran @ IPVS
# 181001

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
from utils.keras_epinet import define_epinet_v1
from utils.mod_epinet import MOD_EPINET
from utils.epinet_dataset import EPI_Data_Loader as Loader
import utils.epinet_dataset as dataset

import matplotlib.pyplot as plt


def set_seed(config):
    seed = config["seed"]
    set_random_seed(seed)
    np.random.seed(seed)
    # if not config["no_cuda"]:
    #     torch.cuda.manual_seed(seed)
    random.seed(seed)

def eval(config):

    epi_model = MOD_EPINET(config)
    # epi_model.model = define_epinet_v1((512,512),5,
    #                                    config['conv_depth'],
    #                                    config['filt_num'],
    #                                    0.1)
    epi_model.summary()
    epi_model.eval()
    epi_model.load_weights(config['input_weights'])
    print(" Finish loading weights!!!!")
    # load images
    # view_n = [1,2,3,4,5,6,7,8,9]
    # in90, in0, in45, in45m = dataset.make_multiinput(config['image_folder'],512,512,view_n)
    loader = Loader(config)
    in0, in90, in45, in45m = loader.get_inputs()
    dis = loader.get_disparity()
    print("size of dis, ", dis.shape)
    # predict
    pre_dis = epi_model.predict([in90, in0, in45, in45m])
    pre_dis = np.squeeze(pre_dis)
    print('size of predicted ',pre_dis.shape)
    gt_dis =dis[11:-11,11:-11]

    mse =np.mean((pre_dis - gt_dis)**2)
    print("mse: ", mse)

    plt.subplot(1,2,1)
    plt.imshow(gt_dis)
    plt.subplot(1,2,2)
    plt.imshow(pre_dis)
    plt.show()
    raw_input()

def eval_old(config):

    epi_model = MOD_EPINET(config)
    epi_model.model = define_epinet_v1((512,512),5,
                                       config['conv_depth'],
                                       config['filt_num'],
                                       0.1)
    epi_model.summary()
    epi_model.eval()
    epi_model.load_weights(config['input_weights'])
    print(" Finish loading weights!!!!")
    # load images
    view_n = [1,2,3,4,5]
    in90, in0, in45, in45m = dataset.make_multiinput(config['image_folder'],512,512,view_n)
    loader = Loader(config)
    # in90, in0, in45, in45m = loader.get_inputs()
    dis = loader.get_disparity()
    print("size of dis, ", dis.shape)
    # predict
    pre_dis = epi_model.predict([in90, in0, in45, in45m])
    pre_dis = np.squeeze(pre_dis)
    print('size of predicted ',pre_dis.shape)
    gt_dis =-1.0*dis[11:-11,11:-11]

    mse =np.mean((pre_dis - gt_dis)**2)
    print("mse: ", mse)

    plt.subplot(1,2,1)
    plt.imshow(gt_dis)
    plt.subplot(1,2,2)
    plt.imshow(pre_dis)
    plt.show()
    raw_input()


def main():
    config = {}

    config['mode'] = 'eval'

    # config['image_folder'] = '/home/trantu/lightfield/local/hci/full/additional/greek'
    config['image_folder'] = '/home/trantu/lightfield/datasets/hci/full/additional/greek'
    # config['disparity_folder'] = '/home/trantu/lightfield/local/hci/full/depths/additional/greek'
    config['disparity_folder'] = '/home/trantu/lightfield/datasets/hci/full/depths/additional/greek'
    config['output_folder'] = './outputs/greek'
    config['patch_size'] = 512
    config['image_height'] = 512
    config['image_width'] = 512
    config['input_weights'] = './logs/view_5_v2/181001_140233/improvement-118-0.22.hdf5'
    # config['input_weights'] = '../epinet/epinet_checkpoints/iter12640_5x5mse1.526_bp5.96.hdf5'

    # config['model_file'] = './logs/view_5_v2/180928_144015/model.json'

    # network config
    config['view_size'] = 5
    config['filt_num'] = 70
    config['conv_depth'] = 7
    config['batch_size'] = 16

    config['seed'] = 1234
    if not os.path.exists(config['output_folder']):
        os.makedirs(config['output_folder'])

    # GPU setting ( gtx 1080ti - gpu0 )
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    set_seed(config)
    if config["mode"] == "eval":
        eval(config)
        # eval_old(config)


if __name__ == "__main__":
    main()
