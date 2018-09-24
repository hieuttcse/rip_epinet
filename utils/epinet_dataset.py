#Trung-Hieu Tran @ IPVS
#180918
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

def make_epiinput(image_path,seq1,image_h,image_w,view_n,RGB):
    traindata_tmp=np.zeros((1,image_h,image_w,len(view_n)),dtype=np.float32)
    i=0
    # if(len(image_path)==1):
    #     image_path=image_path[0]
    for seq in seq1:
        tmp  = np.float32(imageio.imread(image_path+'/input_Cam0%.2d.png' % seq))
        traindata_tmp[0,:,:,i]=(RGB[0]*tmp[:,:,0] + RGB[1]*tmp[:,:,1] + RGB[2]*tmp[:,:,2])/255
        i+=1
    return traindata_tmp


def make_epiinput_lytro(image_path,seq1,image_h,image_w,view_n,RGB):
    traindata_tmp=np.zeros((1,image_h,image_w,len(view_n)),dtype=np.float32)
    i=0
    if(len(image_path)==1):
        image_path=image_path[0]
    for seq in seq1:
        tmp  = np.float32(imageio.imread(image_path+'/%s_%02d_%02d.png' % (image_path.split("/")[-1],1+seq//9, 1+seq-(seq//9)*9)) )
        traindata_tmp[0,:,:,i]=(RGB[0]*tmp[:,:,0] + RGB[1]*tmp[:,:,1] + RGB[2]*tmp[:,:,2])/255
        i+=1
    return traindata_tmp

def make_multiinput(image_path, image_h, image_w, view_n):
    RGB = [0.299, 0.587, 0.114] # RGB to gray

    idx_start= int(0.5*(9-len(view_n)))

    seq90d=list(range(4,81,9)[::-1][idx_start:9-idx_start:])
    # 90degree:  [76, 67, 58, 49, 40, 31, 22, 13, 4 ]
    seq0d=list(range(36,45,1)[idx_start:9-idx_start:])
    # 0degree:  [36, 37, 38, 39, 40, 41, 42, 43, 44]
    seq45d=list(range(8,80,8)[::-1][idx_start:9-idx_start:])
    # 45degree:  [72, 64, 56, 48, 40, 32, 24, 16, 8 ]
    seqM45d=list(range(0,81,10)[idx_start:9-idx_start:])
    # -45degree:  [0, 10, 20, 30, 40, 50, 60, 70, 80]

    if(image_path[:8]=='training' and os.listdir(image_path)[0][:9]=='input_Cam'):
        val_90d=make_epiinput(image_path,seq90d,image_h,image_w,view_n,RGB)
        val_0d=make_epiinput(image_path,seq0d,image_h,image_w,view_n,RGB)
        val_45d=make_epiinput(image_path,seq45d,image_h,image_w,view_n,RGB)
        val_M45d=make_epiinput(image_path,seqM45d,image_h,image_w,view_n,RGB)
    elif(image_path[:5]=='lytro'):
        val_90d=make_epiinput_lytro(image_path,seq90d,image_h,image_w,view_n,RGB)
        val_0d=make_epiinput_lytro(image_path,seq0d,image_h,image_w,view_n,RGB)
        val_45d=make_epiinput_lytro(image_path,seq45d,image_h,image_w,view_n,RGB)
        val_M45d=make_epiinput_lytro(image_path,seqM45d,image_h,image_w,view_n,RGB)

    return val_90d , val_0d, val_45d, val_M45d

def rotation_augmentation(seq_90d_batch, seq_0d_batch,
                          seq_45d_batch,seq_M45d_batch,
                          label_batch, batch_size):
    for batch_i in range(batch_size):
        rot90_rand = np.random.randint(0,4)
        transp_rand = np.random.randint(0,2)
        if transp_rand==1:
            seq_90d_batch_tmp6=np.copy(np.transpose(
                np.squeeze(seq_90d_batch[batch_i,:,:,:]),(1, 0, 2)))
            seq_0d_batch_tmp6=np.copy(np.transpose(
                np.squeeze(seq_0d_batch[batch_i,:,:,:]),(1, 0, 2)))
            seq_45d_batch_tmp6=np.copy(np.transpose(
                np.squeeze(seq_45d_batch[batch_i,:,:,:]),(1, 0, 2)) )
            seq_M45d_batch_tmp6=np.copy(np.transpose(
                np.squeeze(seq_M45d_batch[batch_i,:,:,:]),(1, 0, 2)) )

            seq_0d_batch[batch_i,:,:,:] = seq_90d_batch_tmp6[:,:,::-1]
            seq_90d_batch[batch_i,:,:,:] = seq_0d_batch_tmp6[:,:,::-1]
            seq_45d_batch[batch_i,:,:,:] = seq_45d_batch_tmp6[:,:,::-1]
            seq_M45d_batch[batch_i,:,:,:]= seq_M45d_batch_tmp6[:,:,:]
            label_batch[batch_i,:,:]=np.copy(np.transpose(
                label_batch[batch_i,:,:],(1, 0)))

        if rot90_rand==1: # 90 degree
            seq_90d_batch_tmp3=np.copy(np.rot90(seq_90d_batch[batch_i,:,:,:],1,(0,1)))
            seq_0d_batch_tmp3=np.copy(np.rot90(seq_0d_batch[batch_i,:,:,:],1,(0,1)))
            seq_45d_batch_tmp3=np.copy(np.rot90(seq_45d_batch[batch_i,:,:,:],1,(0,1)))
            seq_M45d_batch_tmp3=np.copy(np.rot90(seq_M45d_batch[batch_i,:,:,:],1,(0,1)))

            seq_90d_batch[batch_i,:,:,:]=seq_0d_batch_tmp3
            seq_45d_batch[batch_i,:,:,:]=seq_M45d_batch_tmp3
            seq_0d_batch[batch_i,:,:,:]=seq_90d_batch_tmp3[:,:,::-1]
            seq_M45d_batch[batch_i,:,:,:]=seq_45d_batch_tmp3[:,:,::-1]
            label_batch[batch_i,:,:]=np.copy(np.rot90(label_batch[batch_i,:,:],1,(0,1)))

        if rot90_rand==2: # 180 degree

            seq_90d_batch_tmp4=np.copy(np.rot90(seq_90d_batch[batch_i,:,:,:],2,(0,1)))
            seq_0d_batch_tmp4=np.copy(np.rot90(seq_0d_batch[batch_i,:,:,:],2,(0,1)))
            seq_45d_batch_tmp4=np.copy(np.rot90(seq_45d_batch[batch_i,:,:,:],2,(0,1)))
            seq_M45d_batch_tmp4=np.copy(np.rot90(seq_M45d_batch[batch_i,:,:,:],2,(0,1)))

            seq_90d_batch[batch_i,:,:,:]=seq_90d_batch_tmp4[:,:,::-1]
            seq_0d_batch[batch_i,:,:,:]=seq_0d_batch_tmp4[:,:,::-1]
            seq_45d_batch[batch_i,:,:,:]=seq_45d_batch_tmp4[:,:,::-1]
            seq_M45d_batch[batch_i,:,:,:]=seq_M45d_batch_tmp4[:,:,::-1]
            label_batch[batch_i,:,:]=np.copy(np.rot90(label_batch[batch_i,:,:],2,(0,1)))

        if rot90_rand==3: # 270 degree
            seq_90d_batch_tmp5=np.copy(np.rot90(seq_90d_batch[batch_i,:,:,:],3,(0,1)))
            seq_0d_batch_tmp5=np.copy(np.rot90(seq_0d_batch[batch_i,:,:,:],3,(0,1)))
            seq_45d_batch_tmp5=np.copy(np.rot90(seq_45d_batch[batch_i,:,:,:],3,(0,1)))
            seq_M45d_batch_tmp5=np.copy(np.rot90(seq_M45d_batch[batch_i,:,:,:],3,(0,1)))
            seq_90d_batch[batch_i,:,:,:]=seq_0d_batch_tmp5[:,:,::-1]
            seq_0d_batch[batch_i,:,:,:]=seq_90d_batch_tmp5
            seq_45d_batch[batch_i,:,:,:]=seq_M45d_batch_tmp5[:,:,::-1]
            seq_M45d_batch[batch_i,:,:,:]=seq_45d_batch_tmp5
            label_batch[batch_i,:,:]=np.copy(np.rot90(label_batch[batch_i,:,:],3,(0,1)))

    return seq_90d_batch, seq_0d_batch, seq_45d_batch, seq_M45d_batch, label_batch


class DatasetType(Enum):
    TRAIN = 0
    DEV = 1
    TEST = 2

class Dataset_Generator(Sequence):
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
        images = self.inputs[istart:iend,...]
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

class EPI_Dataset():
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
    # def patch_extraction(self):
    #     imgs_90d, imgs_0d, imgs_45d, imgs_M45d = make_multiinput(image_path,
    #                                                              512,
    #                                                              512
    #                                                              )
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

    def read_pair_single(self, image_path, disparity_path):
        center = (4,4) # (y, x)
        if self.aug_shift:
            shift_max = (9 - self.view_size)//2
            shift_range = range(-shift_max,shift_max+1)
            yy,xx = np.meshgrid(shift_range,shift_range)
            yy = yy.ravel()
            xx = xx.ravel()
            center_list = [(center[0]+yy[i],center[1]+xx[i]) for i in range(len(xx))]
        else:
            center_list = [center]
        inputs = []
        dis_maps = []

        type_list = [Input_Type.D0, Input_Type.D90,
                     Input_Type.D45, Input_Type.D45M]
        for cen in center_list:
            # reading input images
            adata ={}
            for atype in type_list:
                indexes = self.get_img_numbers(atype, cen,self.view_size)
                images = self.get_epi_input(image_path,indexes,
                                            self.image_height,self.image_width,
                                            self.view_size)
                adata.update({atype: images})
            inputs.append(adata)
            # reading disparity map
            # disparity map was calculated in a reverse order.
            dis = -1.0*self.get_disparity(disparity_path,cen)
            dis_maps.append(dis)
        return inputs, dis_maps
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
            inputs, dis_maps = self.read_pair_single(image_path,disparity_path)
            print(" ... Image Len: ", len(inputs), len(dis_maps))
            #TODO more augmentation strategy
            # convert to numpy for convenience.
            np_inputs = np.zeros([len(inputs),len(Input_Type),
                                  self.image_height,self.image_width,
                                  self.view_size],
                                 dtype=np.uint8)
            np_dis_maps = np.array(dis_maps)
            for i,input in enumerate(inputs):
                for itype in Input_Type:
                    np_inputs[i,itype.value,...] = inputs[i][itype]
            # Cut into image patch
            for idx in range(0,np_inputs.shape[0]):
                # no_patch x no_shifted_view x input_type x patch_size x patch_size x view_size
                patch_inputs, patch_maps = self.prepare_patch(np.expand_dims(np_inputs[idx,...],axis=0),
                                                              np.expand_dims(np_dis_maps[idx,...],axis=0))
                ish = patch_inputs.shape
                patch_inputs.shape = (ish[0]*ish[1],ish[2],ish[3],ish[4],ish[5])
                msh = patch_maps.shape
                patch_maps.shape = (msh[0]*msh[1],msh[2],msh[3])
                # write to file
                n_batch = ish[0]
                print(" ... Write %d patches %d/%d"%(n_batch,idx,np_inputs.shape[0]))
                      #end="\r"0
                self.append_data(patch_inputs,patch_maps)
                del patch_inputs
                del patch_maps
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
