import cv2
import numpy as np
from timeit import default_timer as timer
from numba import vectorize,float64,int32
import scipy
from scipy import ndimage

CCRF_LUT = np.loadtxt('LUTs/CCRF_mono_2x.txt')

@vectorize(["uint8(uint8,uint8)"],target='parallel')
def parallel_CCRF(f2q,fq):
     return CCRF_LUT[fq][f2q]

@vectorize([float64(float64,float64,float64)],target='parallel')
def parallel_pixels(unscaled_output,lower_limit,upper_limit):
    return (unscaled_output - lower_limit) / (upper_limit - lower_limit)

@vectorize([float64(float64,float64,float64)],target='parallel')
def contrast_enhance(pixels, blurred_pixels, intensity):
    x = (pixels - blurred_pixels) * intensity
    return x / (np.abs(x) + 1)


def ldr_tonemap_rgb_image(img, power, radius):
    blured_pixels = ndimage.filters.gaussian_filter(np.mean(img, axis=2), radius)
    intensity = power * power * 0.1

    for channel in range(img.shape[2]):
        pixels = img[:, :, channel]
        #print(pixels.shape)
        unscaled_output = contrast_enhance(pixels, blured_pixels, intensity)

        lower_limit = contrast_enhance(0.0, blured_pixels, intensity)
        upper_limit = contrast_enhance(1.0, blured_pixels, intensity)

        pixels = (unscaled_output - lower_limit) / (upper_limit - lower_limit)

        img[:, :, channel] = pixels
    return img


def ldr_tonemap_rgb_image_cv2(img, power, radius):
    blured_pixels = cv2.GaussianBlur(np.mean(img, axis=2), (11, 11), radius)
    intensity = power * power * 0.1

    for channel in range(img.shape[2]):
        pixels = img[:, :, channel]
        #print(pixels.shape)
        unscaled_output = contrast_enhance(pixels, blured_pixels, intensity)


        lower_limit = contrast_enhance(0.0, blured_pixels, intensity)
        upper_limit = contrast_enhance(1.0, blured_pixels, intensity)

        pixels = (unscaled_output - lower_limit) / (upper_limit - lower_limit)

        img[:, :, channel] = pixels
    return img

def ldr_tonemap_L_image(img, power, radius):
    img = img.reshape(img.shape[0],img.shape[1],1)
    blured_pixels = ndimage.filters.gaussian_filter(np.mean(img, axis=2), radius)
    intensity = power * power * 0.1
    pixels = img[:,:,0]
    unscaled_output = contrast_enhance(pixels, blured_pixels, intensity)

    lower_limit = contrast_enhance(0.0, blured_pixels, intensity)
    upper_limit = contrast_enhance(1.0, blured_pixels, intensity)
    pixels = parallel_pixels(unscaled_output,lower_limit,upper_limit)

    img = pixels
    return img







if __name__ == "__main__":

    #f2q = cv2.imread("5.jpg",cv2.IMREAD_GRAYSCALE)
    #fq = cv2.imread("4.jpg",cv2.IMREAD_GRAYSCALE)
    #parallel_start = timer()
    #ccrf = parallel_CCRF(f2q,fq)
    #parallel_time = timer() - parallel_start
    #print("parallel CCRF Look Up took %f seconds/frame" % parallel_time)
    #cv2.imwrite('6.jpg',ccrf)


    # test cv2

    #img = cv2.cvtColor(cv2.imread('6.jpg',cv2.IMREAD_GRAYSCALE),cv2.COLOR_GRAY2RGB) * (1.0 / 255.0)

    img = cv2.imread('2.jpg',cv2.IMREAD_GRAYSCALE) * (1.0 / 255.0)
    start = timer()
    img = ldr_tonemap_L_image(img, 5, 50)
    end_time = timer() - start
    print("ldr_tonemapping took %f seconds/frame" % end_time)
    scipy.misc.imsave("tono_mapped1.jpg", img * 255.0)
