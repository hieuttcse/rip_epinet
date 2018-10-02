#Trung-Hieu Tran @ IPVS
# 181002
# Reimplement of EPINET
# from keras.utils import Sequence

from __future__ import print_function
import imageio
import numpy as np
import os
from enum import Enum
from keras.utils import Sequence
import pytools.file_io as file_io
import lfutils
import random
from matplotlib import pyplot as plt
import time
import h5py
import random

import io_utils as ioutil

# Load training data from H5 file and feed it to the training model
class Train_Dataset_Loader(Sequence):
    def __init__(self,batch_list, config):
        self.batch_size = config['batch_size']
        h5file = config['input_file']
        batch_size = config['batch_size']
        dset_input = config['dset_input']
        dset_label = config['dset_label']
        self.f = h5py.File(h5file,'r')
        self.batch_list = batch_list
        self.inputs = self.f[dset_input]
        self.labels = self.f[dset_label]

    def __del__(self):
        self.f.close()

    def __len__(self):
        return len(self.batch_list)

    def __getitem__(self,index):
        batch_id = self.batch_list[index]
        istart = batch_id * self.batch_size
        iend = istart + self.batch_size
        images = np.asarray(self.inputs[istart:iend,...],dtype=np.float32)/255.0
        labels = self.labels[istart:iend,...]
        # trim labels
        labels = labels[:,11:-11,11:-11]
        V0 = np.squeeze(images[:,0,...])
        V90 = np.squeeze(images[:,1,...])
        V45 = np.squeeze(images[:,2,...])
        V45M = np.squeeze(images[:,3,...])
        labels = np.expand_dims(labels,axis=-1)
        return [V0,V90,V45,V45M], labels

    @classmethod
    def splits(cls,config):
        h5file = config['input_file']
        batch_size = config['batch_size']
        dset_input = config['dset_input']
        dset_label = config['dset_label']
        with h5py.File(h5file,'r') as f:
            inputs = f[dset_input]
            labels = f[dset_label]
            n_images = labels.shape[0]

        n_batches = int(n_images // batch_size)
        size_test = int(np.floor(n_batches *0.1))
        size_dev = int(np.floor(n_batches * 0.1))
        size_train = n_batches - size_test - size_dev

        batch_list = range(0,n_batches)
        random.shuffle(batch_list)
        train_list = batch_list[0:size_train]
        dev_list = batch_list[size_train:size_train+size_dev]
        test_list = batch_list[-size_test:]

        datasets = (cls(train_list,config),
                    cls(dev_list,config),
                    cls(test_list,config))
        return datasets

class Input_Type(Enum):
    D90 = 0
    D0 = 1
    D45 = 2
    D45M = 3

# reading EPI input data for evaluation, testing purpose
class EPI_Data_Reader():
    def __init__(self,config):
        self.image_folder = config['image_folder']
        self.view_size = config['view_size']
        self.image_height = config['image_height']
        self.image_width = config['image_width']
        if 'disparity_folder' in config:
            self.disparity_folder = config['disparity_folder']
        else:
            self.disaprity_folder = None

    def get_inputs(self):
        type_list = [Input_Type.D0, Input_Type.D90,
                     Input_Type.D45, Input_Type.D45M]
        cen = (4,4)
        adata = {}
        for atype in type_list:
            indexes = self.get_img_numbers(atype, cen, self.view_size)
            print(atype.name," : ",indexes)
            images = self.get_epi_input(self.image_folder,indexes,
                                            self.image_height,self.image_width,
                                            self.view_size)
            images = np.expand_dims(images,axis = 0)
            adata.update({atype: images})
        in90 = np.asarray(adata[Input_Type.D90],dtype=np.float32)/255.0
        # in90 = in90[...,::-1]
        in0  = np.asarray(adata[Input_Type.D0],dtype=np.float32)/255.0
        in45 =  np.asarray(adata[Input_Type.D45],dtype=np.float32)/255.0
        # in45 = in45[...,::-1]
        in45M =  np.asarray(adata[Input_Type.D45M],dtype=np.float32)/255.0

        # return (in90,in0,in45,in45M)
        return (in0,in90,in45,in45M)

    def get_disparity(self):
        cen = (4,4)
        if self.disparity_folder is None:
            return np.zeros((self.image_height,self.image_width))

        position = cen[0]*9 + cen[1]
        fpath = os.path.join(self.disparity_folder,
                             'gt_disp_lowres_Cam0%.2d.pfm'%position)
        dis = -1.0* np.asarray(file_io.read_pfm(fpath),dtype=np.float32)
        return dis

    @classmethod
    def get_epi_input(cls, image_path,indexes, image_height, image_width, view_size):
        RGB = [0.299, 0.587, 0.114] # RGB to gray
        data = np.zeros((image_height,image_width,view_size),dtype=np.float32)
        i=0
        for index in indexes:
            tmp  = np.float32(imageio.imread(image_path+'/input_Cam0%.2d.png' % index))
            data[:,:,i]=(RGB[0]*tmp[:,:,0] + RGB[1]*tmp[:,:,1] + RGB[2]*tmp[:,:,2])
            i+=1
        data = data.astype(np.uint8)
        return data

    @classmethod
    def get_img_indexes(cls,input_type, center, view_size):
        if not isinstance(center, (list,tuple)):
            c_x = center%9
            c_y = center//9
        else:
            c_y = center[0]
            c_x = center[1]
        indexes = []
        half=view_size//2
        x_arr = range(c_x-half, c_x + half+1)
        y_arr = range(c_y-half, c_y + half+1)

        if input_type == Input_Type.D90:
            indexes = [(i_y,c_x) for i_y in y_arr]
        elif input_type == Input_Type.D0:
            indexes = [(c_y,i_x) for i_x in x_arr]
        elif input_type == Input_Type.D45:
            indexes = [(y_arr[i],x_arr[::-1][i]) for i in range(0,view_size)]
        elif input_type == Input_Type.D45M:
            indexes = [(y_arr[i],x_arr[i]) for i in range(0,view_size)]
        return indexes

    @classmethod
    def get_img_numbers(cls,input_type,center,view_size):
        indexes = cls.get_img_indexes(input_type,center,view_size)
        numbers = [ idx[0]*9 + idx[1] for idx in indexes]
        return numbers


# Generating data for training
class Train_Dataset_Generator():
    def __init__(self,config):
        self.data_dir = config['data_dir']
        self.disparity_dir = config['disparity_dir']
        self.patch_size = config['patch_size']
        self.stride = config['stride']

        self.output_file = './out.h5'
        self.input_file = None
        if 'output_file' in config:
            self.output_file = config['output_file']
        if 'input_file' in config:
            self.input_file = config['input_file']
        self.view_size = config['view_size'] # 5, 7, 9
        self.aug_shift = config['aug_shift']
        self.aug_scale = config['aug_scale']
        self.aug_scale_factors = config['aug_scale_factors']
        self.aug_rotate = config['aug_rotate']
        self.thres_patch = config['thres_patch']
        self.dset_input = config['dset_input']
        self.dset_label = config['dset_label']
        self.image_height = 512
        self.image_width = 512

    @classmethod
    def get_disparity(cls, disparity_path, position):
        if isinstance(position,(list,tuple)):
            position = position[0]*9 + position[1]
        fpath = os.path.join(disparity_path,'gt_disp_lowres_Cam0%.2d.pfm'%position)
        dis = np.asarray(file_io.read_pfm(fpath),dtype=np.float32)
        return dis

    @classmethod
    def get_epi_input(cls, image_path,indexes, image_height, image_width, view_size):
        RGB = [0.299, 0.587, 0.114] # RGB to gray
        data = np.zeros((image_height,image_width,view_size),dtype=np.float32)
        i=0
        for index in indexes:
            tmp  = np.float32(imageio.imread(image_path+'/input_Cam0%.2d.png' % index))
            data[:,:,i]=(RGB[0]*tmp[:,:,0] + RGB[1]*tmp[:,:,1] + RGB[2]*tmp[:,:,2])
            i+=1
        data = data.astype(np.uint8)
        return data

    @classmethod
    def get_img_indexes(cls,input_type, center, view_size):
        if not isinstance(center, (list,tuple)):
            c_x = center%9
            c_y = center//9
        else:
            c_y = center[0]
            c_x = center[1]
        indexes = []
        half=view_size//2
        x_arr = range(c_x-half, c_x + half+1)
        y_arr = range(c_y-half, c_y + half+1)

        if input_type == Input_Type.D90:
            indexes = [(i_y,c_x) for i_y in y_arr]
        elif input_type == Input_Type.D0:
            indexes = [(c_y,i_x) for i_x in x_arr]
        elif input_type == Input_Type.D45:
            indexes = [(y_arr[i],x_arr[::-1][i]) for i in range(0,view_size)]
        elif input_type == Input_Type.D45M:
            indexes = [(y_arr[i],x_arr[i]) for i in range(0,view_size)]
        return indexes

    @classmethod
    def get_img_numbers(cls,input_type,center,view_size):
        indexes = cls.get_img_indexes(input_type,center,view_size)
        numbers = [ idx[0]*9 + idx[1] for idx in indexes]
        return numbers

    # Reading input images, gt disparity and applying data augmentation techniques.
    def read_pair_single(self, image_path, disparity_path):
        center = (4,4) # (y, x)
        # if applying shift augmentation, shifting center and reading coresponding images.
        if self.aug_shift:
            shift_max = (9 - self.view_size)//2
            shift_range = range(-shift_max,shift_max+1)
            yy,xx = np.meshgrid(shift_range,shift_range)
            yy = yy.ravel()
            xx = xx.ravel()
            center_list = [(center[0]+yy[i],center[1]+xx[i]) for i in range(len(xx))]
        else:
            center_list = [center]

        type_list = [Input_Type.D0, Input_Type.D90,
                     Input_Type.D45, Input_Type.D45M]
        np_inputs = np.zeros([len(center_list),len(type_list),
                                self.image_height,self.image_width,
                                self.view_size],
                                dtype=np.uint8)
        np_maps = np.zeros([len(center_list),
                            self.image_height,self.image_width],
                           dtype=np.float32)
        for i,cen in enumerate(center_list):
            # reading input images
            for atype in type_list:
                indexes = self.get_img_numbers(atype, cen,self.view_size)
                images = self.get_epi_input(image_path,indexes,
                                            self.image_height,self.image_width,
                                            self.view_size)
                np_inputs[i,atype.value,...] = images[...]
            # reading disparity map
            # disparity map was calculated in a reverse order.
            dis = -1.0*self.get_disparity(disparity_path,cen)
            np_maps[i,...] = dis[...]

        return np_inputs, np_maps
    def good_patch(self,data):
        cen_idx  = int(self.view_size//2)
        apatch = np.squeeze(data[0,0,:,:,cen_idx])
        val = np.mean(np.abs(apatch - np.mean(apatch)))
        if(val<self.thres_patch):
            return False
        return True
    def prepare_patch(self,inputs,dis_maps):
        # calculate number of inputs
        n_x = int(np.ceil((self.image_width-self.patch_size)/self.stride) + 1)
        n_y = int(np.ceil((self.image_height-self.patch_size)/self.stride) + 1)
        # N x 4 x 512 x 512 x 7
        ishape = inputs.shape
        patch_inputs = []
        # vnp.zeros([n_x*n_y,ishape[0],ishape[1],
        #                          self.patch_size,self.patch_size,
        #                          ishape[4]])
        # N x 512 x512
        patch_maps = []
        # np.zeros([n_x*n_y,ishape[0],
        #                        self.patch_size,self.patch_size])
        for ix in range(n_x):
            for iy in range(n_y):
                xstart = ix*self.stride
                ystart = iy*self.stride
                patch = inputs[:,:,
                               ystart:ystart+self.patch_size,
                               xstart:xstart+self.patch_size,:]
                if not self.good_patch(patch):
                    continue
                patch_inputs.append(patch)
                # patch_inputs[iy*n_x + ix,
                #              :,:,:,:,:] = patch
                # patch_maps[iy*n_x + ix,
                           # :,:,:] =
                patch_maps.append(dis_maps[:,
                                           ystart:ystart+self.patch_size,
                                             xstart:xstart+self.patch_size])
        print(" ... ... selected %d of %d patches"%(len(patch_maps),(n_x*n_y)))
        return np.array(patch_inputs), np.array(patch_maps)
    def append_data(self, data, label):
        ioutil.append_data(self.output_file,self.dset_input,data)
        ioutil.append_data(self.output_file,self.dset_label,label)

    def prepare_input_with_augmentation(self,images,maps,scale=1.0,rotate=0):
        print("... Shape ", images.shape)
        # rotation:
        seq_0 = images[Image_]

        # for iidx in range(len(inputs)):
        #     ainput = inputs[iidx]
        #     adis = dis_maps[iidx]
        #     factors = [1.0]
        #     if self.aug_scale:
        #         factors = self.scale_factors
        #     for factor in factors:
        #         arr_input = [ainput]
        #         arr_dis = [adis]
        #         # Scaling augmentation
        #         if not factor == 1.0:
        #             newInput, newDis = self.scaling_augmentation(ainput,adis,scale_factor)
        #             arr_input.append(newInput)
        #             arr_dis.append(newDis)
        #         if self.aug_rotate:
        #             newInputs, newDiss = self.rotating_augementation(ainput,adis)
        #             for i in range(len(newDiss)):
        #                 arr_input.append(newInputs[i])
        #                 arr_dis.append(newDiss[i])
        #         #TODO more augmentation strategy
        #         # convert to numpy for convenience.
        #         np_inputs = np.zeros([len(arr_input),len(Input_Type),
        #                                 self.image_height,self.image_width,
        #                                 self.view_size],
        #                                 dtype=np.uint8)
        #         np_dis_maps = np.array(arr_dis)
        #         for i in range(len(arr_input)):
        #             for itype in Input_Type:
        #                 np_inputs[i,itype.value,...] = arr_input[i][itype]
        #         # Cut into image patch
        #         for idx in range(0,np_inputs.shape[0]):
        #             # no_patch x no_shifted_view x input_type x patch_size x patch_size x view_size
        #             patch_inputs, patch_maps = self.prepare_patch(np.expand_dims(np_inputs[idx,...],axis=0),
        #                                                             np.expand_dims(np_dis_maps[idx,...],axis=0))
        #             ish = patch_inputs.shape
        #             patch_inputs.shape = (ish[0]*ish[1],ish[2],ish[3],ish[4],ish[5])
        #             msh = patch_maps.shape
        #             patch_maps.shape = (msh[0]*msh[1],msh[2],msh[3])
        #             # write to file
        #             n_batch = ish[0]
        #             print(" ... Write %d patches %d/%d"%(n_batch,idx,np_inputs.shape[0]))
        #                     #end="\r"0
        #             self.append_data(patch_inputs,patch_maps)
        #             del patch_inputs
        #             del patch_maps


    def prepare_input(self):
        # check available scenes
        all_files = os.listdir(self.data_dir)
        scenes = [ name for name in all_files if
                   os.path.isdir(os.path.join(self.data_dir,name))]

        print("Found these scenes: ",scenes)
        for scene in scenes:
            print(" Processing the scene %s: "%(scene))
            image_path = os.path.join(self.data_dir,scene)
            disparity_path = os.path.join(self.disparity_dir,scene)
            np_images, np_maps = self.read_pair_single(image_path,disparity_path)
            print(" ... Read input images, len: ", len(np_images), len(np_maps))

            scale_params = [1.0]
            rotate_params = [0]
            if self.aug_scale:
                scale_params = self.aug_scale_factors
            if self.aug_rotate:
                rotate_params = [90, 180, 270]
            for idx in range(len(np_images)):
                for scale_param in scale_params:
                    for rotate_param in rotate_params:
                        print(" ", scale_param, " ", rotate_param)
                        self.prepare_input_with_augmentation(images = np_images[idx,...],
                                                             maps = np_maps[idx,...],
                                                             scale=scale_param,
                                                             rotate=rotate_param)
            # f,(ax1,ax2) = plt.subplots(1,2)
            # plt.show()
            # for _ in range(10):
            #     idx = random.randint(0,n_patches)
            #     dis_map = np.squeeze(patch_maps[idx,:,:])
            #     images = np.squeeze(patch_inputs[idx,0,Input_Type.D45M.value,:,:,:])
            #     ax1.imshow(dis_map)
            #     for i in range(images.shape[-1]):
            #         image = images[:,:,i]
            #         ax2.imshow(np.squeeze(image))
            #         plt.pause(1)
            #     raw_input()

            # for i in range(0,len(inputs)):
            #     print("*"*10)
            #     print("* set ",i)
            #     print("*"*10)
            #     for input_type in Input_Type:
            #         print(" Input TYPE ", input_type.name)
            #         ainput = inputs[i][input_type]
            #         indexes = self.get_img_indexes(input_type,(0,0),self.view_size)
            #         indexes = np.transpose(np.array(indexes),[1,0])
            #         print(indexes)
            #         adis = dis_maps[0]
            #         lfutils.evaluate_disp(ainput,indexes,adis)
