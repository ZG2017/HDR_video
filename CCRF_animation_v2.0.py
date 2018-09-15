# coding: utf-8

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.misc import imresize
import numba as nb
import math

LUT_path = "G:\\ECE516\\HDR team\\1024LUT\\"
im1 = "G:\\ECE516\\HDR team\\ycbcrtest\\input image\\04_32768.jpg"   # dark
im2 = "G:\\ECE516\\HDR team\\ycbcrtest\\input image\\05_65536.jpg"   # bright

@nb.jit([nb.float64[:,:,:](nb.float64[:,:,:,:], nb.float64[:,:,:])],nopython=True,nogil=True,parallel=True)
def image_reconstruction_with_jit(input_images,CCRF): 
    for i in range(input_images.shape[0] - 1):
        for j in range(input_images.shape[0] - 1 - i):
            for y in range(input_images.shape[1]):
                for x in range(input_images.shape[2]):
                    for k in range(input_images.shape[3]):
                        dark_index = math.floor((input_images[j,y,x,k]*1024))  # x
                        bright_index = math.floor((input_images[j+1,y,x,k]*1024))  # y 
                        input_images[j,y,x,k] = CCRF[bright_index,dark_index,k]
    return input_images[0]

class CCRF_animation():
    def __init__(self,
                 im1_file,
                 im2_file,
                 CCRF_path,
                 step,
                 wait):
        self.step = step
        self.wait = wait  # ms
        # image data use to compute
        im1 = (cv2.imread(im1_file).astype(np.float64)+0.5)/256.0
        im2 = (cv2.imread(im2_file).astype(np.float64)+0.5)/256.0
        self.input_images = np.concatenate((np.expand_dims(im1,axis = 0),np.expand_dims(im2,axis = 0)),axis = 0)
        
        # CCRF lookup table
        LUT_files = os.listdir(CCRF_path)
        CCRF_r = np.loadtxt(CCRF_path+LUT_files[0])
        CCRF_g = np.loadtxt(CCRF_path+LUT_files[1])
        CCRF_b = np.loadtxt(CCRF_path+LUT_files[2])
        self.CCRF = np.concatenate((np.expand_dims(CCRF_r.T,axis = 2),np.expand_dims(CCRF_g.T,axis = 2),np.expand_dims(CCRF_b.T,axis = 2)),axis = 2)

        # present on screen (0-255)
        self.CCRF_show = np.expand_dims(imresize(np.loadtxt(CCRF_path+LUT_files[0]).T,(256,256)),axis = 2)
        self.im1_show = imresize(cv2.imread(im1_file),(270, 360))
        self.im2_show = imresize(cv2.imread(im2_file),(270, 360))
        self.im_output_show = imresize((image_reconstruction_with_jit(self.input_images,self.CCRF)*256-0.5).astype(np.uint8),(270, 360))
        self.im_output_show_flatten = self.im_output_show.reshape(97200,3)

        # prepare the frame
        self.bg = np.zeros((1080,1920,3),dtype = np.uint8)
        self.bg += 180

        # change here to change layout!!!-------------------------------------------------------------
        self.CCRF_x,self.CCRF_y = 460,87                 # CCRF's left-top coordinate
        self.im1_x,self.im1_y = 410,377                  # dark image's left-top coordinate
        self.im2_x,self.im2_y = 40,77                    # bright image's left-top coordinate
        self.im_output_x,self.im_output_y = 840,212      # output image's left-top coordinate
        # ---------------------------------------------------------------------------------------------

        self.bg[self.CCRF_y:self.CCRF_y+self.CCRF_show.shape[0],self.CCRF_x:self.CCRF_x+self.CCRF_show.shape[1]] = self.CCRF_show
        self.bg[self.im1_y:self.im1_y+self.im1_show.shape[0],self.im1_x:self.im1_x+self.im1_show.shape[1]] = self.im1_show    
        self.bg[self.im2_y:self.im2_y+self.im2_show.shape[0],self.im2_x:self.im2_x+self.im1_show.shape[1]] = self.im2_show
        #bg[im_output_y:im_output_y+im_output_show.shape[0],im_output_x:im_output_x+im_output_show.shape[1]] = im_output_show
        
    def func(self,step):
        self.step = step * 2

    def animation(self):
        # bigan of animation
        dflatten = 0    # flattened pixel on image
        dx,dy = 0,0     # pixel on image
        bg_for_animation = self.bg.copy()
        blank_output_im = np.zeros((97200,3),dtype = np.uint8)
        blank_output_im += 180
        cv2.namedWindow("Speed Control")
        cv2.createTrackbar("Speed", "Speed Control", 1, 60, self.func)
        while 1:
            bg_show = bg_for_animation.copy()
            # lines on 2 images 
            cv2.line(bg_show,
                     (self.im1_x + dx, self.im1_y),
                     (self.im1_x + dx, self.im1_y + self.im1_show.shape[0] - 1),
                     (0,0,255),
                     1)                    # dark vertical 
            cv2.line(bg_show,
                     (self.im1_x, self.im1_y + dy),
                     (self.im1_x + self.im1_show.shape[1] - 1, self.im1_y + dy),
                     (0,0,255),
                     1)                    # dark horizontal
            cv2.line(bg_show,
                     (self.im2_x + dx, self.im2_y),
                     (self.im2_x + dx, self.im2_y + self.im2_show.shape[0] - 1),
                     (0,0,255),
                     1)                    # bright vertical 
            cv2.line(bg_show,
                     (self.im2_x, self.im2_y + dy),
                     (self.im2_x + self.im2_show.shape[1] - 1, self.im2_y + dy),
                     (0,0,255),
                     1)                    # bright horizontal

            # lines on CCRF
            v_x = self.im1_show[dy-1,dx-1,0]
            v_y = self.im2_show[dy-1,dx-1,0]
            cv2.line(bg_show,
                     (self.CCRF_x + v_x, self.CCRF_y),
                     (self.CCRF_x + v_x, self.CCRF_y + self.CCRF_show.shape[0] - 1),
                     (0,0,255),
                     1)                    # CCRF vertical
            cv2.line(bg_show,
                     (self.CCRF_x, self.CCRF_y + v_y),
                     (self.CCRF_x + self.CCRF_show.shape[1] - 1, self.CCRF_y + v_y),
                     (0,0,255),
                     1)                    # CCRF horizontal
            
            # lines from input images to CCRF
            cv2.line(bg_show,
                     (self.im1_x + dx, self.im1_y + dy),
                     (self.im1_x + dx, self.CCRF_y + self.CCRF_show.shape[0] - 1 + 20),
                     (0,0,255),
                     2)                     # for im1
            cv2.line(bg_show,
                     (self.im1_x + dx, self.CCRF_y + self.CCRF_show.shape[0] - 1 + 20),
                     (self.CCRF_x + v_x, self.CCRF_y + self.CCRF_show.shape[0] - 1 + 20),
                     (0,0,255),
                     2)                     # for im1
            cv2.arrowedLine(bg_show,
                            (self.CCRF_x + v_x, self.CCRF_y + self.CCRF_show.shape[0] - 1 + 20),
                            (self.CCRF_x + v_x, self.CCRF_y + self.CCRF_show.shape[0] - 1),
                            (0,0,255),
                            2,
                            tipLength=0.3)  # for im1
            cv2.line(bg_show,
                     (self.im2_x + dx, self.im2_y + dy),
                     (self.CCRF_x - 20, self.im2_y + dy),
                     (0,0,255),
                     2)                     # for im2
            cv2.line(bg_show,
                     (self.CCRF_x - 20, self.im2_y + dy),
                     (self.CCRF_x - 20, self.CCRF_y + v_y),
                     (0,0,255),
                     2)                     # for im2
            cv2.arrowedLine(bg_show,
                            (self.CCRF_x - 20, self.CCRF_y + v_y),
                            (self.CCRF_x, self.CCRF_y + v_y),
                            (0,0,255),
                            2,
                            tipLength=0.3)  # for im2
            
            # lines from CCRF to output image
            #tmp_x = (self.CCRF_x + v_x) - (self.im_output_x + dx)
            #tmp_y = (self.CCRF_y + v_y) - (self.im_output_y + dy)
            '''
            if tmp_x%2 == 0:
                tmp_x = int(tmp_x/2)
            else:
                tmp_x = int((tmp_x+1)/2)
            
            if tmp_y%2 == 0:
                tmp_y = int(tmp_y/2)
            else:
                tmp_y = int((tmp_y+1)/2)
            '''
            cv2.line(bg_show,
                     (self.CCRF_x + v_x, self.CCRF_y + v_y),
                     (self.im_output_x -1 + dx, self.im_output_y -1 + dy),
                     (0,0,255),
                     2)   
            
            '''
            cv2.line(bg_show,
                     (self.CCRF_x + v_x + abs(tmp_y), self.CCRF_y + v_y - tmp_y),
                     (self.im_output_x + dx - abs(tmp_y), self.CCRF_y + v_y - tmp_y),
                     (0,0,255),
                     2)    
            cv2.line(bg_show,
                     (self.im_output_x + dx - abs(tmp_y), self.CCRF_y + v_y - tmp_y), #self.im_output_y + dy + tmp_y
                     (self.im_output_x -1 + dx, self.im_output_y -1 + dy),
                     (0,0,255),
                     2)   
            '''
            # text
            cv2.putText(bg_show, 
                        text = "Input Frame 2 (Exposure: 2q Ev)", 
                        org = (self.im2_x + 10,self.im2_y + 295), 
                        fontFace = cv2.FONT_HERSHEY_COMPLEX, 
                        fontScale = 0.6, 
                        color = (0,0,0),
                        thickness = 1)    # im2
            cv2.putText(bg_show, 
                        text = "Input Frame 1 (Exposure: q Ev)", 
                        org = (self.im1_x + 15,self.im1_y + 295), 
                        fontFace = cv2.FONT_HERSHEY_COMPLEX, 
                        fontScale = 0.6, 
                        color = (0,0,0),
                        thickness = 1)    # im1
            cv2.putText(bg_show, 
                        text = "CCRF Look-Up Table", 
                        org = (self.CCRF_x + 20,self.CCRF_y - 12), 
                        fontFace = cv2.FONT_HERSHEY_COMPLEX, 
                        fontScale = 0.6, 
                        color = (0,0,0),
                        thickness = 1)    # CCRF
            cv2.putText(bg_show, 
                        text = "Output HDR Frame", 
                        org = (self.im_output_x + 80,self.im_output_y - 12), 
                        fontFace = cv2.FONT_HERSHEY_COMPLEX, 
                        fontScale = 0.6, 
                        color = (0,0,0),
                        thickness = 1)    # output frame
            cv2.putText(bg_show, 
                        text = "How CCRF Look-up Table Generates HDR Frame", 
                        org = (190,40), 
                        fontFace = cv2.FONT_HERSHEY_COMPLEX, 
                        fontScale = 1.1,
                        color = (0,0,0),
                        thickness = 2)    # title
            blank_output_im[0:dflatten] = self.im_output_show_flatten[0:dflatten]
            bg_for_animation[self.im_output_y:self.im_output_y+self.im_output_show.shape[0],self.im_output_x:self.im_output_x+self.im_output_show.shape[1]] = blank_output_im.reshape(270,360,3)
            '''
            if dx == 0:
                bg_for_animation[self.im_output_y + dy, self.im_output_x + dx] = self.im_output_show[dy,dx]
            else:
                bg_for_animation[self.im_output_y + dy, self.im_output_x + dx  - self.step:self.im_output_x + dx] = self.im_output_show[dy,dx -self.step:dx]
            if dx > self.im1_show.shape[1]:
                dx = 0
                dy += 1
                if dy >= self.im1_show.shape[0]:
                    dx,dy = 0,0
                    bg_for_animation = self.bg.copy()
            '''
            dflatten += self.step
            dx = dflatten%360
            dy = math.floor(dflatten/360)
            if dflatten >= 97200:
                dflatten = 0
                bg_for_animation = self.bg.copy()
                blank_output_im = np.zeros((97200,3),dtype = np.uint8)
                blank_output_im += 180
            
            if cv2.waitKey(1) == ord('q'):
                break
            cv2.imshow("How CCRF Look-up Table Works", bg_show)
            cv2.waitKey(self.wait)
        cv2.destroyAllWindows()

def main():
    example = CCRF_animation(im1,im2,LUT_path,1,10)
    example.animation()


main()

