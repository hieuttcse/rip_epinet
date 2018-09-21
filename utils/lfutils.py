# Trung-Hieu Tran @IPVS
# 180919

from __future__ import print_function
import numpy as np
import math
import scipy
import matplotlib.image as mpimg
import scipy.ndimage

def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

def warp_image_with_omega(img,index,omega,hy,hx):
    ny = img.shape[0]
    nx = img.shape[1]
    isColor = False
    if(img.ndim==3):
        isColor = True

    xx, yy = np.meshgrid(np.arange(0, nx), np.arange(0, ny))
    cur_x = xx + index[1] * omega/hx
    cur_y = yy + index[0] * omega/hy
    if isColor:
        outImg = np.zeros([ny,nx,3])
        for c in range(0,3):
            outImg[:,:,c] = scipy.ndimage.interpolation.map_coordinates(img[:,:,c],
                                                                        coordinates=[cur_y, cur_x], order=5,
                                                                        mode='nearest')
    else:
        outImg = scipy.ndimage.interpolation.map_coordinates(img,
                                                             coordinates=[cur_y, cur_x], order=5,
                                                             mode='nearest')
    return outImg

# This function applies warping and check for pixel error
# expect images and indexes are numpy array with
# the last dim is image order
# images: H x W [x 3] x K
# indexes: 2 x K
def evaluate_disp(images, indexes, disp):
    images = np.array(images)*255.0
    indexes = np.array(indexes)
    ref_index = -1
    for i in range(indexes.shape[-1]):
        if indexes[0,i] == 0 and indexes[1,i] == 0:
            ref_index = i
            break
    if ref_index == -1:
        print(" Cannot find the reference view!!!")
        return
    ref_img = np.squeeze(images[...,ref_index])
    err_0 = []
    err_1 = []
    for i in range(indexes.shape[-1]):
        img = np.squeeze(images[...,i])
        idx = np.squeeze(indexes[...,i])
        mabs_0 = np.mean(np.abs(ref_img - img))
        wimg = warp_image_with_omega(img,idx,disp,1.0,1.0)
        mabs_1 = np.mean(np.abs(ref_img - wimg))
        err_0.append(mabs_0)
        err_1.append(mabs_1)

    print(" ... NoWarp error: ",err_0)
    print(" ...   Warp error: ",err_1)

# def warp_lf_with_omega(imgs,indexes,omega,hx,hy):
#     outputs = []
#     IMG0 = imgs[indexes.index([0,0])]
#     ny = IMG0.shape[0]
#     nx = IMG0.shape[1]
#     isColor = False
#     if(IMG0.ndim==3):
#         isColor = True

#     xx, yy = np.meshgrid(np.arange(0, nx), np.arange(0, ny))
#     for i in range(0,len(imgs)):
#         index = indexes[i]
#         inImg = imgs[i]
#         cur_x = xx + index[0] * omega/hx
#         cur_y = yy + index[1] * omega/hy
#         if isColor:
#             outImg = np.zeros([ny,nx,3])
#             for c in range(0,3):
#                 outImg[:,:,c] = scipy.ndimage.interpolation.map_coordinates(inImg[:,:,c],
#                                                                             coordinates=[cur_y, cur_x], order=5,
#                                                                             mode='nearest')
#         else:
#             outImg = scipy.ndimage.interpolation.map_coordinates(inImg,
#                                                                  coordinates=[cur_y, cur_x], order=5,
#                                                                  mode='nearest')
#         outputs.append(outImg)
#     return outputs

# # calculate the gradient,
# # divide by hx,hy to compensate the spatial unit
# def grad_xy(img,hx,hy):
#     imgy, imgx = np.gradient(img)
#     return imgx/hx,imgy/hy

# # def grad_xy_bku(img,hx,hy):
# #     # claculate the gradient of this image.
# #     [ny, nx] = img.shape
# #     img_plus = np.zeros([ny+2,nx+2],dtype=np.float64)
# #     img_minus = np.zeros([ny+2,nx+2],dtype=np.float64)
# #     # horizontal derivative.
# #     img_plus[1:-1,2:] = img
# #     img_minus[1:-1,:-2] = img
# #     img_x = (img_minus - img_plus)/2.0/hx
# #     # vertical derivative
# #     img_plus[2:,1:-1] = img
# #     img_minus[:-2,1:-1] = img
# #     img_y = (img_minus - img_plus)/2.0/hy
# #     return img_x[1:-1,1:-1], img_y[1:-1,1:-1]

# # calculate motion tensor between two images
# def motion_tensor_intensity(f0, f1,delta,hx,hy):
#     # gray constancy assumption only motionensor
#     f0x, f0y = grad_xy(f0,hx,hy)
#     f1x, f1y = grad_xy(f1,hx,hy)
#     fx = (f0x + f1x)/2.0
#     fy = (f0y + f1y)/2.0
#     ft = f1 - f0

#     tmp1 = delta[0] * fx + delta[1] * fy

#     J11 = np.power(tmp1,2)
#     J12 = tmp1*ft
#     J22 = ft * ft

#     return J11,J12,J22

# def motion_tensor_grad(f0, f1,delta,hx,hy):
#     # gradient motiontensor
#     f0x, f0y = grad_xy(f0,hx,hy)
#     f1x, f1y = grad_xy(f1,hx,hy)

#     Jx11,Jx12,Jx22 = motion_tensor_intensity(f0x,f1x,delta,hx,hy)
#     Jy11,Jy12,Jy22 = motion_tensor_intensity(f0y,f1y,delta,hx,hy)
#     J11 = (Jx11 + Jy11)/2.0
#     J12 = (Jx12 + Jy12)/2.0
#     J22 = (Jx22 + Jy22)/2.0

#     return J11,J12,J22

# def motion_tensor_grad_bku(f0, f1,delta,hx,hy):
#     # gradient motiontensor
#     f0x, f0y = grad_xy(f0,hx,hy)
#     f1x, f1y = grad_xy(f1,hx,hy)
#     fx = (f0x + f1x)/2.0
#     fy = (f0y + f1y)/2.0

#     fxx,fxy = grad_xy(fx,hx,hy)
#     fyx,fyy = grad_xy(fy,hx,hy)
#     tmp1 = delta[0] * fxx + delta[1] * fxy
#     tmp2 = delta[0] * fyx + delta[1] * fyy


#     J11 = (np.power(tmp1,2) + np.power(tmp2,2)) /2.0
#     J12 = (tmp1*(f1x -f0x) + tmp2*(f1y-f0y))/2.0
#     J22 = (np.power((f1x -f0x),2) + np.power((f1y-f0y),2))/2.0

#     return J11,J12,J22

# def motion_tensor_lf_color(LF,idxLF,hx,hy,mtype='intensity'):
#     ny = LF[0].shape[0]
#     nx = LF[0].shape[1]

#     J11 = np.zeros([3,ny, nx], dtype=np.float64)
#     J12 = np.zeros([3,ny, nx], dtype=np.float64)
#     J22 = np.zeros([3,ny, nx], dtype=np.float64)

#     IMG0 = LF[idxLF.index([0,0])]
#     count = 0
#     for i in range(0,len(LF)):
#         for index in range(0,3):
#             IMGi = LF[i][:,:,index]
#             idx  = idxLF[i]
#             if ( idx[0]==0 and idx[1]==0):
#                 continue
#             if mtype == 'intensity':
#                 tJ11,tJ12,tJ22 =  motion_tensor_intensity(IMG0[:,:,index], IMGi, idx, hx, hy)
#             else:
#                 tJ11,tJ12,tJ22 =  motion_tensor_grad(IMG0[:,:,index], IMGi, idx, hx, hy)
#             J11[index,:,:] = J11[index,:,:] + tJ11
#             J22[index,:,:] = J22[index,:,:] + tJ22
#             J12[index,:,:] = J12[index,:,:] + tJ12
#             count = count +1
#     # average
#     J11 = J11/count
#     J12 = J12/count
#     J22 = J22/count
#     J11 = np.mean(J11,axis=0)
#     J12 = np.mean(J12,axis=0)
#     J22 = np.mean(J22,axis=0)
#     return J11,J12,J22

# def motion_tensor_lf_rgb(LF,idxLF,hx,hy,mtype='intensity'):
#     ny = LF[0].shape[0]
#     nx = LF[0].shape[1]

#     J11 = np.zeros([3,ny, nx], dtype=np.float64)
#     J12 = np.zeros([3,ny, nx], dtype=np.float64)
#     J22 = np.zeros([3,ny, nx], dtype=np.float64)

#     IMG0 = LF[idxLF.index([0,0])]
#     count = 0
#     for i in range(0,len(LF)):
#         for index in range(0,3):
#             IMGi = LF[i][:,:,index]
#             idx  = idxLF[i]
#             if ( idx[0]==0 and idx[1]==0):
#                 continue
#             if mtype == 'intensity':
#                 tJ11,tJ12,tJ22 =  motion_tensor_intensity(IMG0[:,:,index], IMGi, idx, hx, hy)
#             else:
#                 tJ11,tJ12,tJ22 =  motion_tensor_grad(IMG0[:,:,index], IMGi, idx, hx, hy)
#             J11[index,:,:] = J11[index,:,:] + tJ11
#             J22[index,:,:] = J22[index,:,:] + tJ22
#             J12[index,:,:] = J12[index,:,:] + tJ12
#             count = count +1
#     # average
#     J11 = J11/count
#     J12 = J12/count
#     J22 = J22/count
#     J11 = np.mean(J11,axis=0)
#     J12 = np.mean(J12,axis=0)
#     J22 = np.mean(J22,axis=0)
#     return J11,J12,J22

# # Calculate the lf motion-tensor from a set of input images
# # LF is a list of sub-aperture images (rearrange from 4D LF)
# # idxLF is a list of indi associated with sub-aperture image in LF
# # hx,hy is spatial unit of x,y
# # mtype = 'intensity' : intesity motion-tensor
# # mtype = 'grad' : gradient motion-tensor
# def motion_tensor_lf_grayscale(LF,idxLF,hx,hy,mtype='intensity'):
#     # resolution params
#     ny = LF[0].shape[0]
#     nx = LF[0].shape[1]
#     # initialize motion-tensor element
#     J11 = np.zeros([ny, nx], dtype=np.float64)
#     J12 = np.zeros([ny, nx], dtype=np.float64)
#     J22 = np.zeros([ny, nx], dtype=np.float64)

#     # IMG0 is reference sub-aperture image.
#     IMG0 = LF[idxLF.index([0,0])]
#     count = 0
#     for i in range(0,len(LF)):
#         IMGi = LF[i]
#         idx  = idxLF[i]
#         if ( idx[0]==0 and idx[1]==0):
#             continue
#         if mtype == 'intensity':
#             tJ11,tJ12,tJ22 =  motion_tensor_intensity(IMG0, IMGi, idx, hx, hy)
#         else:
#             tJ11,tJ12,tJ22 =  motion_tensor_grad(IMG0, IMGi, idx, hx, hy)
#         J11 = J11 + tJ11
#         J22 = J22 + tJ22
#         J12 = J12 + tJ12
#         count = count +1
#     # average
#     J11 = J11/count
#     J12 = J12/count
#     J22 = J22/count
#     return J11,J12,J22

# # this function calculate average MSE
# def average_mse_imgs(imgs,indexes):
#     ny = imgs[0].shape[0]
#     nx = imgs[0].shape[1]
#     total = 0
#     IM0 = imgs[indexes.index([0,0])]
#     for i in range(0,len(imgs)):
#         IMi = imgs[i]
#         mse = np.sum(np.square(IMi - IM0)) / (nx * ny*3)
#         total = total + mse
#     return total/(len(imgs)-1)

# def prepare_pyramid_2D(IM, params):
#     noDim = IM.ndim
#     if noDim == 2:
#         color = False
#         [ny,nx] = IM.shape
#     else:
#         color = True
#         [ny,nx,nc] = IM.shape

#     pyramid = []
#     tmpIM = IM
#     pyramid.append(tmpIM)
#     for l in range(1,params['nlevel']):
#         nny = int(math.floor(ny * pow(params['eta'],l)))
#         nnx = int(math.floor(nx * pow(params['eta'],l)))
#         if color:
#             newIM = np.zeros([nny,nnx,nc],dtype=np.float64)
#             for c in range(0,nc):
#                 newIM[:,:,c] = scipy.misc.imresize(tmpIM[:,:,c],[nny,nnx],interp='bicubic',mode='F')
#         else:
#             newIM = scipy.misc.imresize(tmpIM,[nny,nnx],interp='bicubic',mode='F')
#         pyramid.append(newIM)
#         tmpIM = newIM
#     return pyramid

# def rbg_to_gray_LF(LF):
#     IMG = LF[0]
#     if IMG.ndim != 3:
#         raise Exception('Expected RGB input data!')
#     ny = IMG.shape[0]
#     nx = IMG.shape[1]
#     gLF = []
#     for i in range(0,len(LF)):
#         IMG = LF[i]
#         gimg = rgb2gray(IMG)
#         gLF.append(gimg)
#     return gLF

# # This function prepares the multi resolution pyramid for coarse to fine strategy
# def prepare_pyramid_LF(LF, params):
#     IMG = LF[0]
#     if IMG.ndim == 2:
#         [ny,nx] = IMG.shape
#     else:
#         [ny,nx,nc] = IMG.shape
#         pyramid = []
#         #tmpLF = LF
#     pyramid.append(LF)
#     for l in range(1,params['nlevel']):
#         nny = int(math.floor(ny * pow(params['eta'],l)))
#         nnx = int(math.floor(nx * pow(params['eta'],l)))
#         newLF = []
#         for i in range(0,len(LF)):
#             IMG = LF[i]
#             if IMG.ndim == 3:
#                 nc = IMG.shape[2]
#                 tmpIMG = np.zeros([nny,nnx,nc],dtype=np.float64)
#                 for c in range(0,nc):
#                     tmpIMG[:,:,c] = scipy.misc.imresize(IMG[:,:,c],[nny,nnx],interp='bicubic',mode='F')
#             else:
#                 tmpIMG = scipy.misc.imresize(IMG,[nny,nnx],interp='bicubic',mode='F')
#             newLF.append(tmpIMG)
#         pyramid.append(newLF)
#     return pyramid

# # Function to evaluate data cost function
# # given a lightfield motion tensor 
# def eval_costfunc_data_with_tensor(omega,
#                                    Jg11,Jg12,Jg22,
#                                    JG11,JG12,JG22,
#                                    gamma,
#                                    epsilon = 0.0,
#                                    type='l1'):
#     # calculate data cost
#     costg = Jg11*omega*omega + 2*Jg12*omega + Jg22
#     costG = JG11*omega*omega + 2*JG12*omega + JG22
#     if (type=='l1'):
#         costg = np.sqrt(np.abs(costg) + epsilon)
#         costG = np.sqrt(np.abs(costG)+ epsilon)
#     costg = np.sum(np.abs(costg),axis=None)
#     costG = np.sum(np.abs(costG),axis=None)
#     datacost = costg + gamma*costG
#     return datacost

# # Function to evaluate data cost function
# # given list of sub-aperture imegae
# def eval_costfunc_data(omega,
#                        limgs,
#                        linds,
#                        hx,hy,
#                        gamma, epsilon = 0.0,
#                        type='l1'):
#     wlimgs = warp_lf_with_omega(limgs,linds,omega,hx,hy)
#     IMG0 = wlimgs[linds.index([0,0])]
#     total = 0
#     for i in range(0,len(wlimgs)):
#         index = linds[i]
#         if (index == [0,0]):
#             continue
#         IMGi = wlimgs[i]
#         diff = np.square(IMGi - IMG0)
#         if type == 'l1':
#             diff = np.sqrt(diff + epsilon)
#         total += np.sum(diff,axis=None)
#     return total

# # Function to evaluate smoothness cost function
# def eval_costfunc_smooth(omega,
#                          hx,hy,
#                          epsilon = 0.0,
#                          type='l1'):
#     # calculate smoothness cost
#     omega_x, omega_y = grad_xy(omega,hx,hy)
#     regcost = omega_x**2 + omega_y**2
#     if type == 'l1':
#         regcost = np.sqrt(omega_x**2 + omega_y**2 + epsilon)
#     regcost = np.sum(np.abs(regcost),axis=None)
#     return regcost

