# show the how HDR processings

import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from pylab import *
import math
import PIL
import matplotlib.gridspec as gridspec
import operator
from scipy import ndimage
from scipy import misc
import scipy.signal
import cv2
from scipy.optimize import least_squares
from PIL import Image
import time
from scipy.interpolate import interp2d
from scipy.interpolate import RectBivariateSpline
import numba as nb
import statistics

# paths
pathtest1 = "G:\\ECE516\\HDR team\\test\\input image\\"
pathtest1IQR = "G:\\ECE516\\HDR team\\test\\IQR\\"
pathsave1 = "G:\\ECE516\\HDR team\\test\\"

pathtest2 = "G:\\ECE516\\HDR team\\test2\\input image\\"
pathtest2IQR = "G:\\ECE516\\HDR team\\test2\\IQR\\"
pathsave2 = "G:\\ECE516\\HDR team\\test2\\"

pathtest3 = "G:\\ECE516\\HDR team\\test3\\input image\\"
pathtest3IQR = "G:\\ECE516\\HDR team\\test3\\IQR\\"
pathsave3 = "G:\\ECE516\\HDR team\\test3\\"

pathtestycbcr = "G:\\ECE516\\HDR team\\ycbcrtest\\input image\\"
pathtestycbcrIQR = "G:\\ECE516\\HDR team\\ycbcrtest\\IQR\\"
pathsaveycbcr = "G:\\ECE516\\HDR team\\ycbcrtest\\"


# whole HDR image processing
class HDR:
    def __init__(self, 
                 image_path,
                 save_path,
                 k):
        self.k = k
        self.image_path = image_path
        self.save_path = save_path
        self.a = np.array([0,0,0],dtype = np.float64)
        self.c = np.array([0,0,0],dtype = np.float64)
        self.sigma_1_final = np.zeros((3,256),dtype = np.float64)
        self.sigma_2_final = np.zeros((3,256),dtype = np.float64)
        self.argmin_look_up_table = np.zeros((256,256,3),dtype = np.float64)
        self.com_image_total = np.zeros((256,256,3), dtype = np.float64)              # save the total compapramtric graph
        self.com_image_dataset = np.zeros((0,256,256,3), dtype = np.float64)               # save all the comparamtric graphs

    def build_dataset(self):
        # comparametic graph
        image_files = os.listdir(self.image_path)
        number_of_image = len(image_files)
        example_image = imread(self.image_path + image_files[0])
        image_high, image_wide = example_image.shape[0], example_image.shape[1]
        self.image_dataset = np.zeros((0,image_high,image_wide,3), dtype = np.uint8)     # save all the image
        counter = 0

        for i in image_files:
            image_tmp = np.expand_dims(imread(self.image_path + i),axis = 0)
            self.image_dataset = np.concatenate((self.image_dataset,image_tmp))
            if (self.image_dataset.shape[0] - 1) != 0:
                com_image_tmp = np.zeros((256,256,3), dtype = np.uint8)
                for j in range(image_high):
                    for k in range(image_wide):
                        for z in range(3):
                            com_image_tmp[self.image_dataset[self.image_dataset.shape[0] - 1][j,k,z], self.image_dataset[self.image_dataset.shape[0] - 2][j,k,z],z] += 1.0
                com_image_tmp = np.expand_dims(com_image_tmp, axis = 0)
                self.com_image_dataset = np.concatenate((self.com_image_dataset,com_image_tmp))
            counter += 1
            print("completed: %.2f%%"%(counter*100/number_of_image))
        for i in range(self.com_image_dataset.shape[0]):
            self.com_image_total += self.com_image_dataset[i]
        np.save(self.save_path + "compapramtric graph.npy",self.com_image_total)
        
        fig,axes = plt.subplots(2,2, figsize = (10,10), dpi = 300)
        axes[0, 0].axis([0, 255, 0, 255])
        axes[1, 0].axis([0, 255, 0, 255])
        axes[0, 1].axis([0, 255, 0, 255])
        axes[1, 1].axis([0, 255, 0, 255])
        
        axes[0, 0].imshow(self.com_image_total)
        axes[1, 0].imshow(self.com_image_total[:,:,0], cmap = "gray")
        axes[0, 1].imshow(self.com_image_total[:,:,1], cmap = "gray")
        axes[1, 1].imshow(self.com_image_total[:,:,2], cmap = "gray")
        plt.savefig(self.save_path + "Comparagram.jpg")
        plt.show()

        # learning the func
        curve_x_r = []
        curve_x_g = []
        curve_x_b = []

        curve_y_r = []
        curve_y_g = []
        curve_y_b = []

        for i in range(256):
            for j in range(256):
                if self.com_image_total[i][j][0] != 0: 
                    curve_x_r.append(float(j)/255)
                    curve_y_r.append(float(i)/255)
                if self.com_image_total[i][j][1] != 0:   
                    curve_x_g.append(float(j)/255)
                    curve_y_g.append(float(i)/255)
                if self.com_image_total[i][j][2] != 0:   
                    curve_x_b.append(float(j)/255)
                    curve_y_b.append(float(i)/255)

        self.curve_x_r = np.array([curve_x_r])
        self.curve_x_g = np.array([curve_x_g])
        self.curve_x_b = np.array([curve_x_b])
        self.curve_y_r = np.array([curve_y_r])
        self.curve_y_g = np.array([curve_y_g])
        self.curve_y_b = np.array([curve_y_b])
        
    def learning(self,
                 size_of_batch = 1024,
                 lr = 0.00005,
                 epoch = 8000,
                 init_a = 4.0,
                 init_c = 1.0):
        x_dataset_tmp = [self.curve_x_r,self.curve_x_g,self.curve_x_b]
        y_dataset_tmp = [self.curve_y_r,self.curve_y_g,self.curve_y_b]
        for i in range(3):   # R,G,B    
            loss_train = []
            Xdataset = x_dataset_tmp[i].T
            Ydataset = y_dataset_tmp[i].T

            number_of_batchs = math.ceil(Xdataset.shape[0] / size_of_batch)
            x = tf.placeholder(tf.float64, [None, Xdataset.shape[1]])
            y = tf.placeholder(tf.float64, [None, Ydataset.shape[1]])
            a = tf.Variable(initial_value = init_a, dtype = tf.float64)
            c = tf.Variable(initial_value = init_c, dtype = tf.float64)
            
            # model
            y_hat = self.com_func(x,a,c)
            
            # loss function
            #loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = y_hat,labels = y))/ 2
            loss = tf.reduce_sum(tf.pow(y_hat-y, 2)) / (2 * Xdataset.shape[0])
            # optimizer
            optimizer = tf.train.AdamOptimizer(lr).minimize(loss)
            
            init = tf.global_variables_initializer()
            
            input_data,input_label = tf.train.shuffle_batch([Xdataset,Ydataset],
                                                    batch_size = size_of_batch,
                                                    capacity = 2000,
                                                    min_after_dequeue = 1000,
                                                    enqueue_many = True,
                                                    allow_smaller_final_batch = True)
            
            with tf.Session() as sess:
                # training
                sess.run(init)
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess = sess,coord = coord)
                for j in range(epoch):
                    for k in range(number_of_batchs):
                        minibatch_x,minibatch_y = sess.run([input_data,input_label])
                        sess.run(optimizer,feed_dict={x: minibatch_x, 
                                                      y: minibatch_y})
                        loss_train_tmp = sess.run(loss,feed_dict={x: Xdataset, 
                                                                  y: Ydataset})
                    loss_train.append(loss_train_tmp)
                    if (j + 1) % 500 == 0:
                        print(j + 1)
                coord.request_stop()
                coord.join(threads)
                self.a[i] = a.eval()
                self.c[i] = c.eval()
            
            print(self.a[i],self.c[i])
            
            # plot
            xxx = np.linspace(1,epoch,epoch,dtype = np.int16)
            plt.figure()
            ax1 = plt.subplot()
            plt.title("Cross Entrpy Loss on Logistic Regression Model")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            ax1.plot(xxx,loss_train,'r-',label = "Training")
            plt.legend(loc=1,shadow=True)
            plt.savefig(self.save_path + "Logisticloss_%d.jpg"%i)
            plt.show()
            plt.figure(figsize = (3,3))
            plt.plot(Xdataset,Ydataset,",",linspace(0,1,256),self.com_func(linspace(0,1,256),self.a[i],self.c[i]),'r-')
            plt.show()
        np.save(self.save_path + "a_file",self.a)
        np.save(self.save_path + "c_file",self.c)
            
    def com_func(self,f,a,c):
        return (f*self.k**(a*c))/(f**(1/c)*(self.k**a-1)+1)**c
    
    def function(self,q,a,c):    #response function
        '''
        if channel == "r"
            a,c = self.a[0],self.c[0]
        if channel == "g"
            a,c = self.a[1],self.c[1]
        if channel == "b"
            a,c = self.a[2],self.c[2]
        '''
        return 255.0*(q**a/(q**a+1))**c

    def Qfunction(self,q,a,c):
        return 255.0/(1+np.e**(-a*np.log(q)))**c

    def Qcertainty(self,q,a,c):    # certainty
        return (a*c*(np.e**(-a*np.log(q))+1)**(-c)/(np.e**(a*np.log(q))+1))

    def certainty(self,q,a,c):     # qcertainty
        return (a*c*((q**a)/(q**c+1))**c)/(q**(a+1)+q)

    def dfunction(self,q,a,c):
        return c*((a*q**(a-1))/(q**a+1)-(a*q**(2*a-1))/(q**a+1)**2) * (q**a/(q**a+1)**(c-1))
    
    def finverse(self,f,a,c):
        return (((f+0.5)/256)**(1/c)/(1-((f+0.5)/256)**(1/c)))**(1/a)
    
    def plot_printer(self,a,c):
        plt.figure(figsize = (4,4))
        plt.title("Response Function(q)")
        plt.plot(linspace(0.0001,5,10000),self.function(linspace(0.0001,5,10000),a,c),'r-')
        plt.show()
        
        plt.figure(figsize = (4,4))
        plt.title("Certainty Function(q)")
        plt.plot(linspace(0.0001,5,10000),self.certainty(linspace(0.0001,5,10000),a,c),'r-')
        plt.show()

        plt.figure(figsize = (4,4))
        plt.title("Response Function(Q)")
        plt.plot(np.log(linspace(0.0001,5,10000)),self.Qfunction(linspace(0.0001,5,10000),a,c),'r-')
        plt.show()

        plt.figure(figsize = (4,4))
        plt.title("Certainty Function(Q)")
        plt.plot(np.log(linspace(0.0001,5,10000)),self.Qcertainty(linspace(0.0001,5,10000),a,c),'r-')
        plt.show()
        
        plt.figure(figsize = (4,4))
        plt.title("Response Function Inverse(q)")
        plt.plot(linspace(0,255,10000),self.finverse(linspace(0,255,10000),a,c),'r-')
        plt.show()

    def compute_sigma1(self):
        # Sigma1 and Sigma2 vectors
        # com_imagetotal
        tmp_vector  = np.zeros((1,256),dtype = np.float64)
        tmp_add_total = 0.0
        tmp_add_25 = 0.0
        tmp_add_75 = 0.0
        sigma_1 = np.zeros((3,256),dtype = np.float64)    # used to save the sigma_1 trace plot
        sigma_2 = np.zeros((3,256),dtype = np.float64)    # used to save the sigma_2 trace plot

        for z in range(3):
            # compute Sigma_1 vector
            for i in range(256):
                tmp_vector = np.zeros((1,256),dtype = np.float64)
                tmp_add_total = 0.0
                tmp_add_25 = 0.0
                tmp_add_75 = 0.0
                for j in range(256):
                    tmp_vector[0][j] = self.com_image_total[j][i][z]
                tmp_add_total = np.sum(tmp_vector)
                for j in range(256):
                    tmp_add_25 += tmp_vector[0][j]
                    if tmp_add_25 >= tmp_add_total/4.0:
                        sigma_1[z][i] = float(j)
                        break
                for j in range(256):
                    tmp_add_75 += tmp_vector[0][j]
                    if tmp_add_75 >= (3 * tmp_add_total)/4.0:
                        sigma_1[z][i] = float(j) - sigma_1[z][i]
                        if sigma_1[z][i] == 0:
                            sigma_1[z][i] = 1.0
                        break
                sigma_1[z][i] = sigma_1[z][i]/1.349

            # compute Sigma_2 vector
            for i in range(256):
                tmp_vector = np.zeros((1,256),dtype = np.float64)
                tmp_add_total = 0.0
                tmp_add_25 = 0.0
                tmp_add_75 = 0.0
                for j in range(256):
                    tmp_vector[0][j] = self.com_image_total[i][j][z]
                tmp_add_total = np.sum(tmp_vector)
                for j in range(256):
                    tmp_add_25 += tmp_vector[0][j]
                    if tmp_add_25 >= tmp_add_total/4.0:
                        sigma_2[z][i] = float(j)
                        break
                for j in range(256):
                    tmp_add_75 += tmp_vector[0][j]
                    if tmp_add_75 >= (3 * tmp_add_total)/4.0:
                        sigma_2[z][i] = float(j) - sigma_2[z][i]
                        if sigma_2[z][i] == 0:
                            sigma_2[z][i] = 1.0
                        break

                sigma_2[z][i] = sigma_2[z][i]/1.349

        # smoothing  little trick: for sigma_1 make padding to be zeros at the end of array. 
        for z in range(3):
            kernel_size = 40
            conv_core = cv2.getGaussianKernel(kernel_size, sigma = 10).T
            expaned_sigma_1 = np.zeros((1,256 + kernel_size - 1),dtype = np.float64)
            expaned_sigma_2 = np.zeros((1,256 + kernel_size - 1),dtype = np.float64)
            if (kernel_size-1)%2 == 0:
                expaned_sigma_1[0][0:int((kernel_size-1)/2)] = sigma_1[z][0]
                expaned_sigma_1[0][int((kernel_size-1)/2):-int((kernel_size-1)/2)] = sigma_1[z]
                expaned_sigma_1[0][-int((kernel_size-1)/2):256 + kernel_size] = 0

                expaned_sigma_2[0][0:int((kernel_size-1)/2)] = sigma_2[z][0]
                expaned_sigma_2[0][int((kernel_size-1)/2):-int((kernel_size-1)/2)] = sigma_2[z]
                expaned_sigma_2[0][-int((kernel_size-1)/2):256 + kernel_size] = sigma_2[z][255]

            else:
                expaned_sigma_1[0][0:int((kernel_size-2)/2)] = sigma_1[z][0]
                expaned_sigma_1[0][int((kernel_size-2)/2):-int(kernel_size/2)] = sigma_1[z]
                expaned_sigma_1[0][-int(kernel_size/2):256 + kernel_size] = 0

                expaned_sigma_2[0][0:int((kernel_size-2)/2)] = sigma_2[z][0]
                expaned_sigma_2[0][int((kernel_size-2)/2):-int(kernel_size/2)] = sigma_2[z]
                expaned_sigma_2[0][-int(kernel_size/2):256 + kernel_size] = sigma_2[z][255]

            self.sigma_1_final[z] = scipy.signal.convolve(expaned_sigma_1, conv_core, mode= "valid")
            self.sigma_2_final[z] = scipy.signal.convolve(expaned_sigma_2, conv_core, mode= "valid")
            
            '''
            conv_core = cv2.getGaussianKernel(10,5).T
            self.sigma_1_final[z] = scipy.signal.convolve(np.expand_dims(sigma_1[z], axis = 0), conv_core, mode= "same")
            self.sigma_2_final[z] = scipy.signal.convolve(np.expand_dims(sigma_2[z], axis = 0), conv_core, mode= "same")
            '''
            plt.figure(figsize = (8,4))
            plt.title("sigma_1_%d"%z)
            plt.plot(linspace(0,255,256),sigma_1[z],'r-',label = "Original")
            plt.plot(linspace(0,255,256),self.sigma_1_final[z],'b-',label = "Smoothed")
            plt.legend(loc=1,shadow=True)
            plt.show()    

            plt.figure(figsize = (8,4))
            plt.title("sigma_2_%d"%z)
            plt.plot(linspace(0,255,256),sigma_2[z],'r-',label = "Original")
            plt.plot(linspace(0,255,256),self.sigma_2_final[z],'b-',label = "Smoothed")
            plt.legend(loc=2,shadow=True)
            plt.show()

            plt.figure(figsize = (8,4))
            plt.title("Two_sigmas_%d"%z)
            plt.plot(linspace(0,255,256),self.sigma_1_final[z],'b-',label = "Sigma_1")
            plt.plot(linspace(0,255,256),self.sigma_2_final[z],'y-',label = "Sigma_2")
            plt.legend(loc=2,shadow=True)
            plt.savefig(self.save_path + "Sigma_%d.jpg"%z)
            plt.show()
        np.save(self.save_path + "final_sigma_1.npy", self.sigma_1_final)
        np.save(self.save_path + "final_sigma_2.npy", self.sigma_2_final)
        
    def compute_sigma2(self):
        tmp_vector  = np.zeros((1,256),dtype = np.float64)
        tmp_expectation = 0.0
        tmp_deviation = 0.0
        tmp_total_point = 0.0
        tmp_sum = 0.0
        sigma_1 = np.zeros((3,256),dtype = np.float64)    # used to save the sigma_1 trace plot
        sigma_2 = np.zeros((3,256),dtype = np.float64)    # used to save the sigma_2 trace plot

        for z in range(3):
            # compute Sigma_1 vector
            for i in range(256):
                tmp_vector = np.zeros((1,256),dtype = np.float64)
                tmp_expectation = 0.0
                tmp_deviation = 0.0
                tmp_total_point = 0.0
                tmp_sum = 0.0
                for j in range(256):                                  # compute the expectation of one col/row
                    tmp_vector[0][j] = self.com_image_total[j][i][z]
                    tmp_sum += tmp_vector[0][j] * float(j)                  # save the sum of all the data points
                tmp_total_point = np.sum(tmp_vector)
                if tmp_total_point == 0:
                    tmp_total_point = 1.0    # to avoid invalid value
                tmp_expectation = tmp_sum/tmp_total_point
                tmp_sum = 0.0
                for j in range(256):
                    tmp_sum += tmp_vector[0][j] * (float(j) - tmp_expectation)**2   # save the sum of deviation
                tmp_deviation = tmp_sum/tmp_total_point
                sigma_1[z][i] = np.sqrt(tmp_deviation)
                # compute Sigma_2 vector
            for i in range(256):
                tmp_vector = np.zeros((1,256),dtype = np.float64)
                tmp_expectation = 0.0
                tmp_deviation = 0.0
                tmp_total_point = 0.0
                tmp_sum = 0.0
                for j in range(256):                                  # compute the expectation of one col/row
                    tmp_vector[0][j] = self.com_image_total[i][j][z]
                    tmp_sum += tmp_vector[0][j] * float(j)                  # save the sum of all the data points
                tmp_total_point = np.sum(tmp_vector)
                if tmp_total_point == 0:
                    tmp_total_point = 1.0    # to avoid invalid value
                tmp_expectation = tmp_sum/tmp_total_point
                tmp_sum = 0.0
                for j in range(256):
                    tmp_sum += tmp_vector[0][j] * (float(j) - tmp_expectation)**2   # save the sum of deviation
                tmp_deviation = tmp_sum/tmp_total_point
                sigma_2[z][i] = np.sqrt(tmp_deviation)

        # smoothing   little trick: for sigma_1 make padding to be zeros at the end of array. 
        for z in range(3):
            kernel_size = 40
            conv_core = cv2.getGaussianKernel(kernel_size, sigma = 10).T
            expaned_sigma_1 = np.zeros((1,256 + kernel_size - 1),dtype = np.float64)
            expaned_sigma_2 = np.zeros((1,256 + kernel_size - 1),dtype = np.float64)
            if (kernel_size-1)%2 == 0:
                expaned_sigma_1[0][0:int((kernel_size-1)/2)] = sigma_1[z][0]
                expaned_sigma_1[0][int((kernel_size-1)/2):-int((kernel_size-1)/2)] = sigma_1[z]
                expaned_sigma_1[0][-int((kernel_size-1)/2):256 + kernel_size] = 0

                expaned_sigma_2[0][0:int((kernel_size-1)/2)] = sigma_2[z][0]
                expaned_sigma_2[0][int((kernel_size-1)/2):-int((kernel_size-1)/2)] = sigma_2[z]
                expaned_sigma_2[0][-int((kernel_size-1)/2):256 + kernel_size] = sigma_2[z][255]

            else:
                expaned_sigma_1[0][0:int((kernel_size-2)/2)] = sigma_1[z][0]
                expaned_sigma_1[0][int((kernel_size-2)/2):-int(kernel_size/2)] = sigma_1[z]
                expaned_sigma_1[0][-int(kernel_size/2):256 + kernel_size] = 0

                expaned_sigma_2[0][0:int((kernel_size-2)/2)] = sigma_2[z][0]
                expaned_sigma_2[0][int((kernel_size-2)/2):-int(kernel_size/2)] = sigma_2[z]
                expaned_sigma_2[0][-int(kernel_size/2):256 + kernel_size] = sigma_2[z][255]

            self.sigma_1_final[z] = scipy.signal.convolve(expaned_sigma_1, conv_core, mode= "valid")
            self.sigma_2_final[z] = scipy.signal.convolve(expaned_sigma_2, conv_core, mode= "valid")
            for i in range(self.sigma_1_final.shape[1]):
                if self.sigma_1_final[z][i] == 0:
                    self.sigma_1_final[z][i] = 0.0001
                if self.sigma_2_final[z][i] == 0:
                    self.sigma_2_final[z][i] = 0.0001

            '''
            conv_core = cv2.getGaussianKernel(10,5).T
            self.sigma_1_final[z] = scipy.signal.convolve(np.expand_dims(sigma_1[z], axis = 0), conv_core, mode= "same")
            self.sigma_2_final[z] = scipy.signal.convolve(np.expand_dims(sigma_2[z], axis = 0), conv_core, mode= "same")
            '''
            plt.figure(figsize = (8,4))
            plt.title("sigma_1_%d"%z)
            plt.plot(linspace(0,255,256),sigma_1[z],'r-',label = "Original")
            plt.plot(linspace(0,255,256),self.sigma_1_final[z],'b-',label = "Smoothed")
            plt.legend(loc=1,shadow=True)
            plt.show()    

            plt.figure(figsize = (8,4))
            plt.title("sigma_2_%d"%z)
            plt.plot(linspace(0,255,256),sigma_2[z],'r-',label = "Original")
            plt.plot(linspace(0,255,256),self.sigma_2_final[z],'b-',label = "Smoothed")
            plt.legend(loc=2,shadow=True)
            plt.show()

            plt.figure(figsize = (8,4))
            plt.title("Two_sigmas_%d"%z)
            plt.plot(linspace(0,255,256),self.sigma_1_final[z],'b-',label = "Sigma_1")
            plt.plot(linspace(0,255,256),self.sigma_2_final[z],'y-',label = "Sigma_2")
            plt.legend(loc=2,shadow=True)
            plt.savefig(self.save_path + "Sigma_updated_%d.jpg"%z)
            plt.show()
        np.save(self.save_path + "final_sigma_1_updated.npy", self.sigma_1_final)
        np.save(self.save_path + "final_sigma_2_updated.npy", self.sigma_2_final)
        
    def minifunc(self,q,f1,f2,sigma1,sigma2,a,c):
        return ((f1-self.function(q,a,c))**2/(sigma1**2)) + ((f2-self.function(self.k*q,a,c))**2/(sigma2**2))
        
    def compute_CCRF1(self):  # use traverse to compute minimun value
        print("Use traverse to compute minimun value!")
        for z in range(3):
            q_sample = []
            for i in range(256):
                q_sample.append(self.finverse(i,self.a[z],self.c[z]))
            for i in range(256): # f1 
                for j in range(256): # f2 
                    tmp = 0.0
                    result_min = inf
                    q_eastimate = 0.0
                    for q in q_sample:
                        tmp = ((float(i)-self.function(q,self.a[z],self.c[z]))**2/(self.sigma_1_final[z,i]**2)) + ((float(j)-self.function(q*self.k,self.a[z],self.c[z]))**2/(self.sigma_2_final[z,j]**2))
                        if tmp < result_min:
                            result_min = tmp
                            q_eastimate = q
                    self.argmin_look_up_table[j,i,z] = self.function(q_eastimate, self.a[z], self.c[z])
                print("completed: %.2f%%"%(i*100/255.0))
            plt.figure()
            plt.title('argmin lookup table')
            plt.axis([0, 255, 0, 255])
            imshow(self.argmin_look_up_table[:,:,z],cmap = "gray")
            plt.savefig(self.save_path + 'argmin_lookup_table_%d.jpg'%z)
            plt.show()
        self.argmin_look_up_table = self.argmin_look_up_table.astype(np.uint8)
        np.save(self.save_path + "argmin_lookup_table.npy",self.argmin_look_up_table)
        
    def compute_CCRF2(self): # use Levenberg–Marquardt algorithm to compute minimun value
        print("Use Levenberg–Marquardt algorithm to compute minimun value!")
        for z in range(3):
            q_sample = []
            for i in range(256):
                q_sample.append(self.finverse(i,self.a[z],self.c[z]))
            for i in range(256): # f1
                for j in range(256): # f2
                    tmp = least_squares(self.minifunc,self.finverse(0,self.a[z],self.c[z]),args =(float(i),float(j),self.sigma_1_final[z,i],self.sigma_2_final[z,j],self.a[z],self.c[z]))
                    f_eastimate = self.function(tmp.x, self.a[z], self.c[z])
                    if f_eastimate > 255.0:
                        f_eastimate = 255.0
                    elif f_eastimate < 0.0:
                        f_eastimate = 0
                    self.argmin_look_up_table[j,i,z] = f_eastimate
                print("completed: %.2f%%"%(i*100/255.0))
            plt.figure()
            plt.title('argmin lookup table')
            plt.axis([0, 255, 0, 255])
            imshow(self.argmin_look_up_table[:,:,z],cmap = "gray")
            #plt.savefig(self.save_path + 'argmin_lookup_table_%d.jpg'%z)
            plt.show()
        self.argmin_look_up_table = self.argmin_look_up_table.astype(np.uint8)
        #np.save(self.save_path + "argmin_lookup_table.npy",self.argmin_look_up_table)

        
    def image_reconstruction(self,image_path):
        image_file = os.listdir(image_path)
        example_image = imread(image_path + image_file[0])
        image_high, image_wide = example_image.shape[0], example_image.shape[1]
        number_of_image = len(image_file)
        tmp_image_input = np.zeros((0,image_high,image_wide,3), dtype = np.uint8)
        
        # prepare the input images
        for i in image_file:
            image_tmp = np.expand_dims(imread(image_path + i),axis = 0)
            tmp_image_input = np.concatenate((tmp_image_input,image_tmp))

        for z in range(number_of_image - 1):
            tmp_image_output = np.zeros((number_of_image - 1 - z ,image_high,image_wide,3), dtype = np.uint8)
            for x in range(number_of_image - 1 - z):
                tmp_image_1 = tmp_image_input[x]
                tmp_image_2 = tmp_image_input[x+1]
                for i in range(image_high):
                    for j in range(image_wide):
                        for k in range(3):
                            tmp_image_output[x][i][j][k] = self.argmin_look_up_table[tmp_image_2[i][j][k]][tmp_image_1[i][j][k]][k]
            tmp_image_input = tmp_image_output.copy()
            print("Layer completed!")
            
        image_output = tmp_image_output[0]

        plt.figure(figsize = (10,10))
        plt.imshow(image_output)
        plt.savefig(self.save_path + 'reconstruction_image.jpg')
        plt.show()
        
    def image_reconstruction2(self,image_1_path,image_2_path):   # compute two pictures
        image_1 = imread(image_1_path)
        image_2 = imread(image_2_path)
        
        image_output = np.empty_like(image_2,dtype = np.uint8)
        
        for i in range(image_output.shape[0]):
            for j in range(image_output.shape[1]):
                for k in range(3):
                    image_output[i,j,k] = self.argmin_look_up_table[image_2[i][j][k],image_1[i][j][k],k]
                    
        image_output = image_output.astype(np.uint8)
        plt.figure(figsize = (10,10))
        imshow(image_output)
        #plt.savefig(self.save_path + 'reconstruction_image.jpg')
        plt.show()
        

    def load_a_c(self,
                 mode,
                 a,
                 c):
        if mode == 0:
            self.a = np.load(a)
            self.c = np.load(c)
        elif mode == 1:
            self.a = a
            self.c = c
        else:
            return "NO SUCH MODE!"
    
    def load_CCRF(self,CCRF_file):
        self.argmin_look_up_table = np.load(CCRF_file)
        
    def load_compapramtric_graph(self,compapramtric_graph_file):
        self.com_image_total = np.load(compapramtric_graph_file)



# example of using HDR class(k = 2)
T = HDR(image_path = pathtest1, save_path = pathsave1, k = 2)
T.build_dataset()
T.learning(size_of_batch = 2048, lr = 0.00006, epoch = 2500, init_a = 12.0, init_c = 0.03)
T.compute_sigma2()
T.compute_CCRF1()

T = HDR(image_path = pathtest1, save_path = pathtest1IQR, k = 2)
T.load_compapramtric_graph("G:\\ECE516\\HDR team\\test\\compapramtric graph.npy")
T.load_a_c(mode = 0, a = "G:\\ECE516\\HDR team\\test\\a_file.npy", c = "G:\\ECE516\\HDR team\\test\\c_file.npy")
T.compute_sigma1()
T.compute_CCRF1()

# example of using HDR class(k = 4)
T = HDR(image_path = pathtest2, save_path = pathsave2, k = 4)
T.build_dataset()
T.learning(size_of_batch = 2048, lr = 0.00006, epoch = 2500, init_a = 8.0, init_c = 0.03)
T.compute_sigma2()
T.compute_CCRF1()

T = HDR(image_path = pathtest2, save_path = pathtest2IQR, k = 4)
T.load_compapramtric_graph("G:\\ECE516\\HDR team\\test2\\compapramtric graph.npy")
T.load_a_c(mode = 0, a = "G:\\ECE516\\HDR team\\test2\\a_file.npy", c = "G:\\ECE516\\HDR team\\test2\\c_file.npy")
T.compute_sigma1()
T.compute_CCRF1()


# example of using HDR class(k = 8)
"""
T = HDR(image_path = pathtest3, save_path = pathsave3, k = 8)
T.build_dataset()
T.learning(size_of_batch = 2048, lr = 0.00006, epoch = 4000, init_a = 4.0, init_c = 0.03)
T.compute_sigma2()
T.compute_CCRF1()

T = HDR(image_path = pathtest3, save_path = pathtest3IQR, k = 8)
T.load_compapramtric_graph("G:\\ECE516\\HDR team\\test3\\compapramtric graph.npy")
T.load_a_c(mode = 0, a = "G:\\ECE516\\HDR team\\test3\\a_file.npy", c = "G:\\ECE516\\HDR team\\test3\\c_file.npy")
T.compute_sigma1()
T.compute_CCRF1()
"""

# tune mapping processing
def tone_mapping_Durand(img_file,
                        gamma = 4, 
                        contrast = 12,
                        saturation = 6,
                        sigma_space = 1.5,
                        sigma_color = 1.5):
            
    '''
    tonemapDrago = cv2.createTonemapDrago(2, 2)
    ldrDrago = tonemapDrago.process(image_output)
    ldrDrago = 3 * ldrDrago
    ldrDrago = ldrDrago * 256.0 - 0.5

    tonemapDurand = cv2.createTonemapDurand(8,4,1.0,1,1)
    ldrDurand = tonemapDurand.process(image_output)
    ldrDurand = 3 * ldrDurand

    tonemapMantiuk = cv2.createTonemapMantiuk(2.2,0.85, 1.2)
    ldrMantiuk = tonemapMantiuk.process(image_output)
    ldrMantiuk = 3 * ldrMantiuk
    
    tonemapReinhard = cv2.createTonemapReinhard(gamma,       
                                                intensity,       # [-8, 8]
                                                light_adapt,     # [0, 1]
                                                color_adapt)     # [0, 1]
    ldrReinhard = tonemapReinhard.process(img)
    ldrReinhard = ldrReinhard * 255.0
    ldrReinhard = ldrReinhard.astype(np.uint8)
    '''
    img = imread(img_file).astype(np.float32)/255
    tonemapDurand = cv2.createTonemapDurand(gamma,
                                            contrast,
                                            saturation,
                                            sigma_space,
                                            sigma_color)
    
    init_time = time.time()
    ldrDurand = tonemapDurand.process(img)
    ldrDurand = ldrDurand * 255.0
    print("passed time = {}".format(time.time()-init_time))
    np.putmask(ldrDurand, ldrDurand > 255, 255)
    np.putmask(ldrDurand, ldrDurand < 0, 0)
    ldrDurand = ldrDurand.astype(np.uint8)

    
    plt.figure(figsize = (10,10))
    plt.imshow(ldrDurand)
    misc.imsave("G:\\ECE516\\HDR team\\ldrDurand.jpg",ldrDurand)
    plt.show()
    
def tone_mapping_Reinhard(img_file,
                          gamma = 1.5,       
                          intensity = 0,       
                          light_adapt = 0,     
                          color_adapt = 0):
            
    '''
    tonemapDrago = cv2.createTonemapDrago(2, 2)
    ldrDrago = tonemapDrago.process(image_output)
    ldrDrago = 3 * ldrDrago
    ldrDrago = ldrDrago * 256.0 - 0.5

    tonemapDurand = cv2.createTonemapDurand(8,4,1.0,1,1)
    ldrDurand = tonemapDurand.process(image_output)
    ldrDurand = 3 * ldrDurand

    tonemapMantiuk = cv2.createTonemapMantiuk(2.2,0.85, 1.2)
    ldrMantiuk = tonemapMantiuk.process(image_output)
    ldrMantiuk = 3 * ldrMantiuk
    '''
    img = imread(img_file).astype(np.float32)/255
    tonemapReinhard = cv2.createTonemapReinhard(gamma,       
                                                intensity,       # [-8, 8]
                                                light_adapt,     # [0, 1]
                                                color_adapt)     # [0, 1]
    
    init_time = time.time()
    ldrReinhard = tonemapReinhard.process(img)
    ldrReinhard = ldrReinhard * 255.0
    np.putmask(ldrReinhard, ldrReinhard > 255, 255)
    np.putmask(ldrReinhard, ldrReinhard < 0, 0)
    ldrReinhard = ldrReinhard.astype(np.uint8)
    print("passed time = {}".format(time.time()-init_time))

    
    plt.figure(figsize = (10,10))
    plt.imshow(ldrReinhard)
    misc.imsave("G:\\ECE516\\HDR team\\Reinhard.jpg",ldrReinhard)
    plt.show()


# example of using tuning mapping func
"""
tone_mapping_Durand("G:\\ECE516\\HDR team\\before_tone.jpg",
                    gamma = 4, 
                    contrast = 20,
                    saturation = 5,
                    sigma_space = 0,
                    sigma_color = 0)

tone_mapping_Reinhard("G:\\ECE516\\HDR team\\before_tone.jpg",
                      gamma = 1.5,       
                      intensity = -3,       
                      light_adapt = 0,     
                      color_adapt = 0)
"""