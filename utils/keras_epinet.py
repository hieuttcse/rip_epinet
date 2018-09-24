# Trung-Hieu Tran @ IPVS
# 180918
# Reimplementation of this model with keras

import keras
import tensorflow as tf
from keras.optimizers import RMSprop
from keras.models import Model, Sequential
from keras.layers import Input, Activation
from keras.layers import Conv2D, Reshape
from keras.layers import Dropout, BatchNormalization
from keras.layers import concatenate


def layer1_multistream(input_dims, filt_num):
    seq = Sequential()

    for i in range(3):
        seq.add(Conv2D(int(filt_num),(2,2),input_shape=input_dims,
                padding='valid', name='S1_c1%d'%(i)))
        seq.add(Activation('relu',name='S1_relu1%d'%(i)))
        seq.add(Conv2D(int(filt_num),(2,2),padding='valid',
                       name='S1_c2%d'%(i)))
        seq.add(BatchNormalization(axis=-1, name='S1_BN%d'%(i)))
        seq.add(Activation('relu', name='S1_relu2%d'%(i)))
    #seq.add(Reshape((int(input_dims[0])-6,int(input_dims[1])-6,int(filt_num))))
    return seq

def layer2_merged(input_dims, filt_num, conv_depth):
    seq = Sequential()
    for i in range(conv_depth):
        seq.add(Conv2D(filt_num,(2,2), padding='valid',
                       input_shape=input_dims, name='S2_c1%d' % (i) ))
        seq.add(Activation('relu', name='S2_relu1%d' %(i))) 
        seq.add(Conv2D(filt_num,(2,2), padding='valid', name='S2_c2%d' % (i))) 
        seq.add(BatchNormalization(axis=-1, name='S2_BN%d' % (i)))
        seq.add(Activation('relu', name='S2_relu2%d' %(i)))
    return seq

def layer3_last(input_dims, filt_num):
    seq = Sequential()
    for i in range(1):
        seq.add(Conv2D(filt_num,(2,2),padding='valid',
                       input_shape=input_dims,name='S3_c1%d'%(i)))
        seq.add(Activation('relu',name='S3_relu1%d'%(i)))
    seq.add(Conv2D(1,(2,2),padding='valid', name='S3_last'))
    return seq

def define_epinet_v1(sz_input, view_size, conv_depth, filt_num, learning_rate):
    input_stack_90d = Input(shape=(sz_input[0],sz_input[1],view_size),
                            name = 'input_stack_90d')
    input_stack_0d = Input(shape=(sz_input[0],sz_input[1],view_size),
                            name = 'input_stack_0d')
    input_stack_45d = Input(shape=(sz_input[0],sz_input[1],view_size),
                            name = 'input_stack_45d')
    input_stack_M45d = Input(shape=(sz_input[0],sz_input[1],view_size),
                            name = 'input_stack_M45d')

    mid_90d= layer1_multistream((sz_input[0],sz_input[1],view_size),
                                int(filt_num))(input_stack_90d)
    mid_0d= layer1_multistream((sz_input[0],sz_input[1],view_size),
                                int(filt_num))(input_stack_0d)
    mid_45d= layer1_multistream((sz_input[0],sz_input[1],view_size),
                                int(filt_num))(input_stack_45d)
    mid_M45d= layer1_multistream((sz_input[0],sz_input[1],view_size),
                                int(filt_num))(input_stack_M45d)

    # Merge layers
    mid_merged = concatenate([mid_90d, mid_0d, mid_45d, mid_M45d],name='mid_merged')
    mid_merged_=layer2_merged((sz_input[0]-6,sz_input[1]-6,int(4*filt_num)),
                              int(4*filt_num),conv_depth)(mid_merged)
    ''' Last Conv layer : Conv - Relu - Conv '''
    output=layer3_last((sz_input[0]-20,sz_input[1]-20,int(4*filt_num)),
                       int(4*filt_num))(mid_merged_)

    model_512 = Model(inputs=[input_stack_90d,input_stack_0d,
                              input_stack_45d,input_stack_M45d],
                      outputs=[output])
    return model_512


def define_epinet(sz_input, view_n, conv_depth, filt_num, learning_rate):
    input_stack_90d = Input(shape=(sz_input[0],sz_input[1],len(view_n)),
                            name = 'input_stack_90d')
    input_stack_0d = Input(shape=(sz_input[0],sz_input[1],len(view_n)),
                            name = 'input_stack_0d')
    input_stack_45d = Input(shape=(sz_input[0],sz_input[1],len(view_n)),
                            name = 'input_stack_45d')
    input_stack_M45d = Input(shape=(sz_input[0],sz_input[1],len(view_n)),
                            name = 'input_stack_M45d')

    mid_90d= layer1_multistream((sz_input[0],sz_input[1],len(view_n)),
                                int(filt_num))(input_stack_90d)
    mid_0d= layer1_multistream((sz_input[0],sz_input[1],len(view_n)),
                                int(filt_num))(input_stack_0d)
    mid_45d= layer1_multistream((sz_input[0],sz_input[1],len(view_n)),
                                int(filt_num))(input_stack_45d)
    mid_M45d= layer1_multistream((sz_input[0],sz_input[1],len(view_n)),
                                int(filt_num))(input_stack_M45d)

    # Merge layers
    mid_merged = concatenate([mid_90d, mid_0d, mid_45d, mid_M45d],name='mid_merged')
    mid_merged_=layer2_merged((sz_input[0]-6,sz_input[1]-6,int(4*filt_num)),
                              int(4*filt_num),conv_depth)(mid_merged)
    ''' Last Conv layer : Conv - Relu - Conv '''
    output=layer3_last((sz_input[0]-20,sz_input[1]-20,int(4*filt_num)),
                       int(4*filt_num))(mid_merged_)

    model_512 = Model(inputs=[input_stack_90d,input_stack_0d,
                              input_stack_45d,input_stack_M45d],
                      outputs=[output])
    return model_512


def main():
    Setting02_AngualrViews = [0,1,2,3,4,5,6,7,8] 
    mod = define_epinet((64,64),Setting02_AngualrViews,
                        7,70,0.1**5)
    mod.summary()
    # for layer in mod.layers:
    #     print(layer.output_shape)

if __name__ =="__main__":
    main()


