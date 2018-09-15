import numpy as np
import os
import matplotlib.pyplot as plt
from pylab import *
import math
import PIL
import matplotlib.gridspec as gridspec
from scipy import ndimage
from scipy import misc
import cv2
from PIL import Image
import time
from scipy.interpolate import interp2d
from scipy.interpolate import RectBivariateSpline
import numba as nb

# image reconstruction with total for-loop
def image_reconstruction(image_path,LUT_path):  # use interp2d
    image_file = os.listdir(image_path)
    LUT_files = os.listdir(LUT_path)
    
    argmin_look_up_table_r = np.loadtxt(LUT_path+LUT_files[0])
    #argmin_look_up_table_r = np.flip(argmin_look_up_table_r,axis=1)
    argmin_look_up_table_g = np.loadtxt(LUT_path+LUT_files[1])
    #argmin_look_up_table_g = np.flip(argmin_look_up_table_g,axis=1)
    argmin_look_up_table_b = np.loadtxt(LUT_path+LUT_files[2])
    #argmin_look_up_table_b = np.flip(argmin_look_up_table_b,axis=1)
    
    plt.figure()
    plt.title('argmin lookup table')
    plt.axis([0, 1024, 0, 1024])
    imshow(argmin_look_up_table_r,cmap = "gray")
    
    plt.show()
    
    plt.figure()
    plt.title('argmin lookup table')
    plt.axis([0, 1024, 0, 1024])
    imshow(argmin_look_up_table_g,cmap = "gray")
    plt.show()
    
    plt.figure()
    plt.title('argmin lookup table')
    plt.axis([0, 1024, 0, 1024])
    imshow(argmin_look_up_table_b,cmap = "gray")
    plt.show()
    
    example_image = imread(image_path + image_file[0])
    image_high, image_wide = example_image.shape[0], example_image.shape[1]
    number_of_image = len(image_file)
    tmp_image_input = np.zeros((0,image_high,image_wide,3), dtype = np.float32)
    
    # interpolation
    xxx = np.arange(0,1,1/1024)
    yyy = np.arange(0,1,1/1024)

    func_r = interp2d(xxx,yyy,argmin_look_up_table_r,kind = "linear")
    func_g = interp2d(xxx,yyy,argmin_look_up_table_g,kind = "linear")
    func_b = interp2d(xxx,yyy,argmin_look_up_table_b,kind = "linear")
    
    func = [func_r,func_g,func_b]

    # prepare the input images
    for i in image_file:
        image_tmp = ((np.expand_dims(imread(image_path + i),axis = 0)+0.5)/256.0).astype(np.float32)
        tmp_image_input = np.concatenate((tmp_image_input,image_tmp))

    for z in range(number_of_image - 1):
        tmp_image_output = np.zeros((number_of_image - 1 - z ,image_high,image_wide,3), dtype = np.float32)
        for x in range(number_of_image - 1 - z):
            tmp_image_1 = tmp_image_input[x]
            tmp_image_2 = tmp_image_input[x+1]
            for i in range(image_high):
                for j in range(image_wide):
                    for k in range(3):
                        tmp_image_output[x][i][j][k] = func[k](tmp_image_2[i][j][k],tmp_image_1[i][j][k])
        tmp_image_input = tmp_image_output.copy()
        print("Layer completed!")

    image_output = tmp_image_output[0]
    
    #misc.imsave("G:\\ECE516\\HDR team\\before_tone.jpg",image_output)
    return image_output


# prepare CCRF table, input image and RectBivariateSpline function
def image_reconstruction_prepare(image_path,LUT_path):
    image_file = os.listdir(image_path)
    LUT_files = os.listdir(LUT_path)
    
    argmin_look_up_table_r = np.loadtxt(LUT_path+LUT_files[0],dtype = np.float32).T
    #argmin_look_up_table_r = np.flip(argmin_look_up_table_r,axis=1)
    argmin_look_up_table_g = np.loadtxt(LUT_path+LUT_files[1],dtype = np.float32).T
    #argmin_look_up_table_g = np.flip(argmin_look_up_table_g,axis=1)
    argmin_look_up_table_b = np.loadtxt(LUT_path+LUT_files[2],dtype = np.float32).T
    #argmin_look_up_table_b = np.flip(argmin_look_up_table_b,axis=1)
    CCRF = np.concatenate((np.expand_dims(argmin_look_up_table_r,axis = 0),np.expand_dims(argmin_look_up_table_g,axis = 0),np.expand_dims(argmin_look_up_table_b,axis = 0)),axis = 0)
    
    plt.figure()
    plt.title('argmin lookup table')
    plt.axis([0, 1024, 0, 1024])
    imshow(argmin_look_up_table_r,cmap = "gray")
    plt.show()
    
    plt.figure()
    plt.title('argmin lookup table')
    plt.axis([0, 1024, 0, 1024])
    imshow(argmin_look_up_table_g,cmap = "gray")
    plt.show()
    
    plt.figure()
    plt.title('argmin lookup table')
    plt.axis([0, 1024, 0, 1024])
    imshow(argmin_look_up_table_b,cmap = "gray")
    plt.show()
    
    # interpolation
    xxx = np.arange(0,1,1/1024)
    yyy = np.arange(0,1,1/1024)
    func_r = RectBivariateSpline(xxx,yyy,argmin_look_up_table_r, bbox = [0,1,0,1])
    func_g = RectBivariateSpline(xxx,yyy,argmin_look_up_table_g, bbox = [0,1,0,1])
    func_b = RectBivariateSpline(xxx,yyy,argmin_look_up_table_b, bbox = [0,1,0,1])
    func = [func_r.ev,func_g.ev,func_b.ev]
    
    # prepare the input images
    example_image = imread(image_path + image_file[0])
    image_high, image_wide = example_image.shape[0], example_image.shape[1]
    tmp_image_input = np.zeros((0,image_high,image_wide,3), dtype = np.float32)
    
    for i in image_file:
        image_tmp = ((np.expand_dims(imread(image_path + i),axis = 0)+0.5)/256.0).astype(np.float32)
        tmp_image_input = np.concatenate((tmp_image_input,image_tmp))
    
    return tmp_image_input,func,CCRF


# image reconstruction with RectBivariateSpline method with 3 CCRF lookup table
def image_reconstruction2(input_images,func):  
    tmp_image_input = input_images
    number_of_image,image_high,image_wide =tmp_image_input.shape[0],tmp_image_input.shape[1],tmp_image_input.shape[2]
    for z in range(number_of_image - 1):
        tmp_image_output = np.zeros((number_of_image - 1 - z ,image_high,image_wide,3), dtype = np.float32)
        for x in range(number_of_image - 1 - z):
            for k in range(3):
                tmp_image_1_channel = tmp_image_input[x][:,:,k].flatten()
                tmp_image_2_channel = tmp_image_input[x+1][:,:,k].flatten()
                tmp_image_output[x][:,:,k] = func[k](tmp_image_1_channel,tmp_image_2_channel).reshape(image_high, image_wide)
        tmp_image_input = tmp_image_output.copy()
        print("Layer completed!")
    #misc.imsave("G:\\ECE516\\HDR team\\before_tone.jpg",image_output)
    return tmp_image_output[0]


# image reconstruction with RectBivariateSpline method with 1 CCRF lookup table (faster)
def image_reconstruction3(image_path,LUT_path): # use 1 CCRF with RectBivariateSpline
    image_file = os.listdir(image_path)
    LUT_files = os.listdir(LUT_path)
    
    argmin_look_up_table_r = np.loadtxt(LUT_path+LUT_files[0])
    #argmin_look_up_table_r = np.flip(argmin_look_up_table_r,axis=1)
    argmin_look_up_table_g = np.loadtxt(LUT_path+LUT_files[1])
    #argmin_look_up_table_g = np.flip(argmin_look_up_table_g,axis=1)
    argmin_look_up_table_b = np.loadtxt(LUT_path+LUT_files[2])
    #argmin_look_up_table_b = np.flip(argmin_look_up_table_b,axis=1)
    
    plt.figure()
    plt.title('argmin lookup table')
    plt.axis([0, 1024, 0, 1024])
    imshow(argmin_look_up_table_r,cmap = "gray")
    plt.show()
    
    plt.figure()
    plt.title('argmin lookup table')
    plt.axis([0, 1024, 0, 1024])
    imshow(argmin_look_up_table_g,cmap = "gray")
    plt.show()
    
    plt.figure()
    plt.title('argmin lookup table')
    plt.axis([0, 1024, 0, 1024])
    imshow(argmin_look_up_table_b,cmap = "gray")
    plt.show()
    
    example_image = imread(image_path + image_file[0])
    image_high, image_wide = example_image.shape[0], example_image.shape[1]
    number_of_image = len(image_file)
    tmp_image_input = np.zeros((0,image_high,image_wide,3), dtype = np.float32)
    
    # interpolation
    xxx = np.arange(0,1,1/1024)
    yyy = np.arange(0,1,1/1024)

    func_r = RectBivariateSpline(xxx,yyy,argmin_look_up_table_r, bbox = [0,1,0,1])
    #func_g = RectBivariateSpline(xxx,yyy,argmin_look_up_table_g, bbox = [0,1,0,1])
    #func_b = RectBivariateSpline(xxx,yyy,argmin_look_up_table_b, bbox = [0,1,0,1])
    
    #func = [func_r.ev,func_g.ev,func_b.ev]
    # prepare the input images
    for i in image_file:
        image_tmp = ((np.expand_dims(imread(image_path + i),axis = 0)+0.5)/256.0).astype(np.float32)
        tmp_image_input = np.concatenate((tmp_image_input,image_tmp))

    init_time = time.time()
    for z in range(number_of_image - 1):
        tmp_image_output = np.zeros((number_of_image - 1 - z ,image_high,image_wide,3), dtype = np.float32)
        for x in range(number_of_image - 1 - z):
                tmp_image_1 = tmp_image_input[x].flatten()
                tmp_image_2 = tmp_image_input[x+1].flatten()
                tmp_image_output[x] = func_r.ev(tmp_image_1,tmp_image_2).reshape(image_high, image_wide,3)
        tmp_image_input = tmp_image_output.copy()
        print("Layer completed!")
    print("final passed time = {}".format(time.time()-init_time))

    image_output = tmp_image_output[0]
    
    #misc.imsave("G:\\ECE516\\HDR team\\before_tone.jpg",image_output)
    return image_output


# vectorize computation speed up
@nb.vectorize([nb.float32(nb.float32, nb.float32, nb.float32, nb.float32, nb.float64, nb.float64)],target='parallel')
def compute_with_vectorize(q11,q21,q12,q22,x,y):
    dark_index = math.floor(x)
    bright_index = math.floor(y)
    return q11*(dark_index+1-x)*(bright_index+1-y)+\
           q21*(x-dark_index)*(bright_index+1-y)+\
           q12*(dark_index+1-x)*(y-bright_index)+\
           q22*(x-dark_index)*(y-bright_index)


# guvectorize computation speed up (basicly equal to vectorize computation)
@nb.guvectorize([(nb.float32[:,:,:,:],nb.float32[:,:,:],nb.float32[:,:,:])],'(i,j,k,m),(m,z,z)->(j,k,m)',nopython=True)
def image_reconstruction_with_guvectorize(input_images,CCRF,output): 
    for i in range(input_images.shape[0] - 1):
        for j in range(input_images.shape[0] - 1 - i):
            for y in range(input_images.shape[1]):
                for x in range(input_images.shape[2]):
                    for k in range(input_images.shape[3]):
                        dark_index = math.floor((input_images[j,y,x,k]*1024))  # x
                        bright_index = math.floor((input_images[j+1,y,x,k]*1024))  # y 
                        q11,q21,q12,q22 = CCRF[k,bright_index,dark_index], CCRF[k,bright_index,dark_index+1], CCRF[k,bright_index+1,dark_index], CCRF[k,bright_index+1,dark_index+1]
                        input_images[j,y,x,k] = q11*(dark_index+1-input_images[j,y,x,k]*1024)*(bright_index+1-input_images[j+1,y,x,k]*1024)+\
                                                q21*(input_images[j,y,x,k]*1024-dark_index)*(bright_index+1-input_images[j+1,y,x,k]*1024)+\
                                                q12*(dark_index+1-input_images[j,y,x,k]*1024)*(input_images[j+1,y,x,k]*1024-bright_index)+\
                                                q22*(input_images[j,y,x,k]*1024-dark_index)*(input_images[j+1,y,x,k]*1024-bright_index)
    output = input_images[0]


# jit computation, even faster
@nb.jit([nb.float32[:,:,:](nb.float32[:,:,:,:], nb.float32[:,:,:])],nopython=True,nogil=True,parallel=True)
def image_reconstruction_with_jit(input_images,CCRF): 
    for i in range(input_images.shape[0] - 1):
        for j in range(input_images.shape[0] - 1 - i):
            for y in range(input_images.shape[1]):
                for x in range(input_images.shape[2]):
                    for k in range(input_images.shape[3]):
                        dark_index = math.floor((input_images[j,y,x,k]*1024))  # x
                        bright_index = math.floor((input_images[j+1,y,x,k]*1024))  # y 
                        q11,q21,q12,q22 = CCRF[k,bright_index,dark_index], CCRF[k,bright_index,dark_index+1], CCRF[k,bright_index+1,dark_index], CCRF[k,bright_index+1,dark_index+1]
                        input_images[j,y,x,k] = q11*(dark_index+1-input_images[j,y,x,k]*1024)*(bright_index+1-input_images[j+1,y,x,k]*1024)+\
                                                q21*(input_images[j,y,x,k]*1024-dark_index)*(bright_index+1-input_images[j+1,y,x,k]*1024)+\
                                                q12*(dark_index+1-input_images[j,y,x,k]*1024)*(input_images[j+1,y,x,k]*1024-bright_index)+\
                                                q22*(input_images[j,y,x,k]*1024-dark_index)*(input_images[j+1,y,x,k]*1024-bright_index)
    return input_images[0]


# following test is the time of combine 6 normal frames to 1 HDR frame

# totally for-loop: 546.1561679840088s
# RectBivariateSpline method with 3 CCRF lookup table: 14.841824769973755s (0.2225947380065918s/channel)
# jit: roughly 0.3s (0.062163591384887695 per combining 3 frames) 
# guvectorize: roughly 0.3s (0.062216997146606445 per combining 3 frames)











