# Trung-Hieu Tran @ IPVS
# 180928
# Reimplementation of this model with keras
# Some modification on the network structure

import keras
import tensorflow as tf
from keras.utils.vis_utils import plot_model

from keras.optimizers import RMSprop
from keras.models import Model, Sequential
from keras.layers import Input, Activation
from keras.layers import Conv2D, Reshape
from keras.layers import Dropout, BatchNormalization
from keras.layers import concatenate

from keras.models import model_from_json
import json


class MOD_EPINET():
    def __init__(self,config=None):
        if config is None:
            return

        self.view_size = config['view_size']
        self.patch_size = config['patch_size']
        if not 'input_height' in config:
            self.input_height = self.patch_size
        else:
            self.input_height = config['input_height']

        if not 'input_width' in config:
            self.input_width = self.patch_size
        else:
            self.input_width = config['input_width']

        self.input_shape = (self.input_height,self.input_width,
                            self.view_size)
        self.filt_num = config['filt_num']
        self.conv_depth = config['conv_depth']
        self.build_model()
    def layer1_multistream(self,prefix,prev_input):
        filter_sizes = [70,50,70]
        x = prev_input
        for i in range(3):
            fsize = filter_sizes[i]
            x = Conv2D(int(fsize),(2,2),padding='valid',
                       name='S1_%s_c1_%d'%(prefix,i))(x)
            x = Activation('relu',name='S1_%s_relu1_%d'%(prefix,i))(x)
            x = Conv2D(int(fsize),(2,2),padding='valid',
                        name='S1_%s_c2_%d'%(prefix,i))(x)
            x = BatchNormalization(axis=-1, name='S1_%s_BN_%d'%(prefix,i))(x)
            x = Activation('relu', name='S1_%s_relu2_%d'%(prefix,i))(x)
        out = x
        return out

    def layer2_merged(self,prev_input):
        fsize = 4*self.filt_num
        x = prev_input
        for i in range(self.conv_depth):
            x = Conv2D(int(fsize),(2,2), padding='valid',
                       name='S2_c1%d' % (i) )(x)
            x = Activation('relu', name='S2_relu1%d' %(i))(x)
            x = Conv2D((int(fsize)),(2,2),
                       padding='valid', name='S2_c2%d' % (i))(x)
            x = BatchNormalization(axis=-1, name='S2_BN%d' % (i))(x)
            x = Activation('relu', name='S2_relu2%d' %(i))(x)
        out = x
        return out

    def layer3_last(self,prev_input):
        fsize = 4*self.filt_num
        x = Conv2D(int(fsize),(2,2),padding='valid',
                   name='S3_c1_0')(prev_input)
        x = Activation('relu',name='S3_relu1_0')(x)
        out = Conv2D(1,(2,2),padding='valid', name='S3_last')(x)
        return out

    def build_model(self):
        input_stack_90d = Input(shape=self.input_shape,
                            name = 'input_stack_90d')
        input_stack_0d = Input(shape=self.input_shape,
                            name = 'input_stack_0d')
        input_stack_45d = Input(shape=self.input_shape,
                            name = 'input_stack_45d')
        input_stack_M45d = Input(shape=self.input_shape,
                            name = 'input_stack_M45d')

        mid_90d  = self.layer1_multistream('90d',input_stack_90d)
        mid_0d   = self.layer1_multistream('0d',input_stack_0d)
        mid_45d  = self.layer1_multistream('45d',input_stack_45d)
        mid_M45d = self.layer1_multistream('M45d',input_stack_M45d)

        # Merge layers
        mid_merged = concatenate([mid_90d, mid_0d, mid_45d, mid_M45d],name='mid_merged')
        mid_merged_= self.layer2_merged(mid_merged)

        # last layer
        output=self.layer3_last(mid_merged_)

        self.model = Model(inputs=[input_stack_90d,input_stack_0d,
                              input_stack_45d,input_stack_M45d],
                      outputs=[output])

    def summary(self):
        self.model.summary()

    def save_weights(self,filename):
        self.model.save_weights(filename)
    def eval(self):
        # configPro = tf.ConfigProto(device_count={'GPU':2,'CPU':2})
        sess = tf.Session()
        keras.backend.set_session(sess)
        keras.backend.get_session().run(tf.global_variables_initializer())

    def predict(self, data_input):
        with self.graph.as_default():
            return self.model.predict(data_input)

    def load_weights(self,filename):
        self.model.load_weights(filename)
        self.graph = tf.get_default_graph()
    def save_model(self,filename):
        model_json = self.model.to_json()
        mjson = json.loads(model_json)
        with open(filename,'w') as f:
            f.write(json.dumps(mjson,indent=3))

    def load_model(self,filename):
        with open(filename,'r') as f:
            jstring = f.read()
            self.model = model_from_json(jstring)
    def export_plot(self,filename,show_shapes=True,
                    show_layer_names=True):
        plot_model(self.model,to_file=filename,
                    show_shapes=show_shapes,
                    show_layer_names=show_layer_names)


def main():
    config = {}
    config['filt_num'] = 70
    config['conv_depth'] = 7
    config['view_size'] = 5
    config['patch_size'] = 23
    mod = MOD_EPINET(config)
    mod.summary()
    mod.export_plot('./model.pdf')
    # for layer in mod.layers:
    #     print(layer.output_shape)

if __name__ =="__main__":
    main()


