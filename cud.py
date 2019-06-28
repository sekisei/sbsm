import numpy as np
from PIL import Image
import glob,os
from matplotlib import pyplot
from numba import cuda
from numba import jit,prange
from tqdm import tqdm

pixel_num_max = 3648*2736
grad = 256

histgram=np.zeros((pixel_num_max,grad),dtype=np.uint16)

@cuda.jit    
def sbsm_gpu(I,hist,hist_sum,out):
    bIdx = cuda.blockIdx.x #36..width
    bIdy = cuda.blockIdx.y #27..height
    dIdx = cuda.blockDim.x

    P_w0 = 0.9
    P_w1 = 0.1
    #Indexs of image are opposite to Indexs about hist array. 
    P_I_given_by_w0 = hist[bIdx][bIdy][I[bIdy][bIdx]]/hist_sum
    P_I_given_by_w1 = 1.0/255.0
    P_I = P_w0*P_I_given_by_w0+P_w1*P_I_given_by_w1
    P_w0_given_by_I = (P_I_given_by_w0*P_w0)/P_I
    P_w1_given_by_I = (P_I_given_by_w1*P_w1)/(P_w0*P_I_given_by_w0+P_w1*P_I_given_by_w1)
    if P_w1_given_by_I > P_w0_given_by_I :
        out[bIdy][bIdx][0] = 255
        out[bIdy][bIdx][1] = 0
        out[bIdy][bIdx][2] = 0
    
if __name__ == '__main__':

    print('loading...')
    histgram = np.load('hist_xy.npy')
    #All sum is same as sample image num.<-important
    hist_sum = np.sum(histgram[0][0],axis=0)
    print('loaded.')
    
    img = Image.open('/home/kai/sbsm/IMAG0760.jpg')
    gray_img = img.convert(mode='L')
    img_array = np.asarray(img)
    gray_img_array = np.asarray(gray_img)
    img_array.flags.writeable = True
    gray_img_array.flags.writeable = True
    sbsm_gpu[(img.width,img.height),1](gray_img_array,histgram,hist_sum,img_array)
    img_out = Image.fromarray(np.uint8(img_array))
    img_out.save('sbsm_gpu.jpg')
    
    
