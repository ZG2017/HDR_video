import cv2
import numpy as np
#from PIL import Image
import sys
from timeit import default_timer as timer
from numba import vectorize


#in use
'''
CCRF_LUT_R_4 = np.loadtxt('LUTs\CCRF_R_4.txt')
CCRF_LUT_G_4 = np.loadtxt('LUTs\CCRF_G_4.txt')
CCRF_LUT_B_4 = np.loadtxt('LUTs\CCRF_B_4.txt')

CCRF_LUT_R_8 = np.loadtxt('LUTs\CCRF_R_8.txt')
CCRF_LUT_G_8 = np.loadtxt('LUTs\CCRF_G_8.txt')
CCRF_LUT_B_8 = np.loadtxt('LUTs\CCRF_B_8.txt')
'''
CCRF_LUT_R_2 = np.loadtxt('LUTs\CCRF_R_2.txt')
CCRF_LUT_G_2 = np.loadtxt('LUTs\CCRF_G_2.txt')
CCRF_LUT_B_2 = np.loadtxt('LUTs\CCRF_B_2.txt')

CCRF_LUT_R_4 = np.loadtxt('LUTs\CCRF_R_4.txt')
CCRF_LUT_G_4 = np.loadtxt('LUTs\CCRF_G_4.txt')
CCRF_LUT_B_4 = np.loadtxt('LUTs\CCRF_B_4.txt')

CCRF_LUT_R_8 = np.loadtxt('LUTs\CCRF_R_8.txt')
CCRF_LUT_G_8 = np.loadtxt('LUTs\CCRF_G_8.txt')
CCRF_LUT_B_8 = np.loadtxt('LUTs\CCRF_B_8.txt')

CCRF_LUT_R_16 = np.loadtxt('LUTs\CCRF_R_16.txt')
CCRF_LUT_G_16 = np.loadtxt('LUTs\CCRF_G_16.txt')
CCRF_LUT_B_16 = np.loadtxt('LUTs\CCRF_B_16.txt')
'''
#NOT in use
#HDR_LUT_R = np.loadtxt('HDR_R.txt')
#HDR_LUT_G = np.loadtxt('HDR_G.txt')
#HDR_LUT_B = np.loadtxt('HDR_B.txt')


'''
'''
@vectorize(["uint8(uint8,uint8)"],target='parallel')
def parallel_HDR_B(f2q_B,fq_B):
     return HDR_LUT_B[fq_B][f2q_B]

@vectorize(["uint8(uint8,uint8)"],target='parallel')
def parallel_HDR_G(f2q_G,fq_G):
     return HDR_LUT_G[fq_G][f2q_G]

@vectorize(["uint8(uint8,uint8)"],target='parallel')
def parallel_HDR_R(f2q_R,fq_R):
     return HDR_LUT_R[fq_R][f2q_R]

#NOT IN USE
def run_CCRF(f2q,fq):
    CCRF = np.zeros(fq.shape)
    for i in range(f2q.shape[0]):
        for j in range(f2q.shape[1]):
            X_B = fq[i][j][0]
            Y_B = f2q[i][j][0]
            CCRF[i][j][0] = CCRF_LUT_B[X_B][Y_B]
            X_G = fq[i][j][1]
            Y_G = f2q[i][j][1]
            CCRF[i][j][1] = CCRF_LUT_G[X_G][Y_G]
            X_R = fq[i][j][2]
            Y_R = f2q[i][j][2]
            CCRF[i][j][2] = CCRF_LUT_R[X_R][Y_R]
    return CCRF
'''
###############k=2 LUT
@vectorize(["uint8(uint8,uint8)"],target='parallel')
def parallel_CCRF_B_2(f2q_B,fq_B):
     return CCRF_LUT_B_2[fq_B][f2q_B]

@vectorize(["uint8(uint8,uint8)"],target='parallel')
def parallel_CCRF_G_2(f2q_G,fq_G):
     return CCRF_LUT_G_2[fq_G][f2q_G]

@vectorize(["uint8(uint8,uint8)"],target='parallel')
def parallel_CCRF_R_2(f2q_R,fq_R):
     return CCRF_LUT_R_2[fq_R][f2q_R]
#################k=4 LUT
@vectorize(["uint8(uint8,uint8)"],target='parallel')
def parallel_CCRF_B_4(f2q_B,fq_B):
     return CCRF_LUT_B_4[fq_B][f2q_B]

@vectorize(["uint8(uint8,uint8)"],target='parallel')
def parallel_CCRF_G_4(f2q_G,fq_G):
     return CCRF_LUT_G_4[fq_G][f2q_G]

@vectorize(["uint8(uint8,uint8)"],target='parallel')
def parallel_CCRF_R_4(f2q_R,fq_R):
     return CCRF_LUT_R_4[fq_R][f2q_R]

################k=8 LUT
@vectorize(["uint8(uint8,uint8)"],target='parallel')
def parallel_CCRF_B_8(f2q_B,fq_B):
     return CCRF_LUT_B_8[fq_B][f2q_B]

@vectorize(["uint8(uint8,uint8)"],target='parallel')
def parallel_CCRF_G_8(f2q_G,fq_G):
     return CCRF_LUT_G_8[fq_G][f2q_G]

@vectorize(["uint8(uint8,uint8)"],target='parallel')
def parallel_CCRF_R_8(f2q_R,fq_R):
     return CCRF_LUT_R_8[fq_R][f2q_R]

#################k=16 LUT
@vectorize(["uint8(uint8,uint8)"],target='parallel')
def parallel_CCRF_B_16(f2q_B,fq_B):
     return CCRF_LUT_B_16[fq_B][f2q_B]

@vectorize(["uint8(uint8,uint8)"],target='parallel')
def parallel_CCRF_G_16(f2q_G,fq_G):
     return CCRF_LUT_G_16[fq_G][f2q_G]

@vectorize(["uint8(uint8,uint8)"],target='parallel')
def parallel_CCRF_R_16(f2q_R,fq_R):
     return CCRF_LUT_R_16[fq_R][f2q_R]
if __name__ == "__main__":

    f2q = cv2.imread("4000.jpeg")
    fq = cv2.imread("2000.jpeg")
    parallel_start = timer()
    f2q_B = f2q[:,:,0]
    f2q_G = f2q[:,:,1]
    f2q_R = f2q[:,:,2]
    fq_B = fq[:,:,0]
    fq_G = fq[:,:,1]
    fq_R = fq[:,:,2]
    ccrf_B = parallel_CCRF_B(f2q_B,fq_B)
    ccrf_G = parallel_CCRF_G(f2q_G,fq_G)
    ccrf_R = parallel_CCRF_R(f2q_R,fq_R)
    ccrf = np.dstack((ccrf_B, ccrf_G, ccrf_R))
    parallel_time = timer() - parallel_start
    print("parallel CCRF Look Up took %f seconds/frame" % parallel_time)
    cv2.imwrite('ccrf.jpeg',ccrf)
    regular_start = timer()
    ccrf = run_CCRF(f2q,fq)
    regular_time = timer() - regular_start
    print("regular CCRF Look Up took %f seconds/frame" % regular_time)