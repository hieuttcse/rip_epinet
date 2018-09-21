# Trung-Hieu Tran @ IPVS
# 180919

from utils.epinet_dataset import  EPI_Dataset
import os

def main():
    config = {}
    config['data_dir'] ='/home/trantu/lightfield/datasets/hci/full/additional'
    config['disparity_dir'] = '/home/trantu/lightfield/datasets/hci/full/depths/additional'

    # config['data_dir'] ='/home/trantu/lightfield/local/hci/full/additional'
    # config['disparity_dir'] = '/home/trantu/lightfield/local/hci/full/depths/additional'
    config['patch_size'] = 23
    config['stride'] = 6
    config['output_file'] = '/home/trantu/maps/pool1/data/epinet/train_7v_23p_6s.h5'
    # config['output_file'] = '/home/trantu/tmp/t9.h5'
    config['view_size'] = 9

    config['aug_shift'] = True
    config['thres_patch'] = 0.03*255 # threshold to select good patch
    config['dset_input'] = 'inputs'
    config['dset_label'] = 'labels'

    if os.path.exists(config['output_file']):
        os.remove(config['output_file'])
    epi_dataset = EPI_Dataset(config)
    # tt = epi_dataset.get_img_indexes(Input_Type.D45,51)
    epi_dataset.prepare_input()
    # print(tt)

if __name__ == "__main__":
    main()
