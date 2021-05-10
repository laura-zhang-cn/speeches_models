# -*- coding: utf-8 -*-
"""
Created on Wed May  5 09:43:37 2021

@author: yxzha

倒谱法提取帧的共振峰特征

"""
import os

import numpy as npy
from scipy.signal import lfilter
from matplotlib import pyplot as plt
import pandas as pds

import  digital_waveform  as dgwv
import waveform_cut as wvct

def local_maxium(x):
    """
    求序列的极大值
    :param x:
    :return:
    """
    d = npy.diff(x)
    l_d = len(d)
    maxium = []
    loc = []
    for i in range(l_d - 1):
        if d[i] > 0 and d[i + 1] <= 0:
            maxium.append(x[i + 1])
            loc.append(i + 1)
    return maxium, loc

''' 倒谱法估计 共振峰'''
def cepstrum_formant(x,winsize=10):
    '''
    使用共振峰模型提取声学特征
    spec包络线 val共振峰幅值 loc共振峰位置 
    '''
    #winsize=10 #int(framerate/10.0*0.2)
    fftx=npy.fft.fft(x)  # 时域变频域
    fftlog=npy.log(npy.abs(fftx[:len(x)//2]))
    ifftx=npy.fft.ifft(fftlog)  # 频域变回时域
    
    pst=npy.zeros(len(ifftx),dtype=npy.complex)
    pst[:winsize]=ifftx[:winsize]
    pst[-winsize+1:]=ifftx[-winsize+1:]
    
    spec=npy.real(npy.fft.fft(pst))  # npy.fft.rfft(pst) # 返回实数的频域值
    val, loc = local_maxium(spec)
    return spec,val,loc

if __name__=='__main__':
    # 读取
    file_path='E:/algorithm_data/speaches_wav_type/'
    fl_names=os.listdir(file_path)
    flx='床前明月光，疑是地上霜.wav' # 夜来风雨声，花落知多少
    speech_text=flx.split('.')[0] #话
    wave_data,nframes,nchannels,framerate= dgwv.get_wav_frame(file_name=file_path+flx)
    
    # 预处理
    tm=pds.Series(npy.arange(0,nframes)*1.0/framerate) # x轴的采样时间  time
    wave_data=wave_data-npy.mean(wave_data) # 去中心化
    wvd=pds.Series(wave_data * 1.0 / (max(abs(wave_data)))) #归一
    
    #
    plt.figure(figsize=(12,5))
    plt.plot(wvd,'b')
    plt.title(speech_text)
    plt.grid()
    plt.show()
    
    # 自定义两个区间，观察观察结果
    plt.figure(figsize=(12,12))
    ixs=[40000,50000]
    k=1
    for ix in ixs:
        wvdx=wvd[ix:ix+1000].reset_index(drop=True) # index重置了
        wvd_lf=lfilter([1, -0.99], [1],wvdx)# 预加重
        #wvd_pw=npy.power(wvdx*10,2)/100*dgwv.sign_num(wvdx) # 预加重，两个效果差不多
        spec,val,loc= cepstrum_formant(wvd=wvd_lf,winsize=10)
        print(len(loc))
        n=2
        plt.subplot(n,3,k)
        plt.plot(wvd_lf,'b')
        plt.title('预加重后波形 {0}'.format(ix))
        #plt.plot(wvd_pw,'r*')
        plt.subplot(n,3,(k+1,k+2))
        plt.plot(spec,'g-')
        plt.plot(loc,val,'r*')
        plt.title('共振峰估计')
        plt.suptitle(speech_text)
        k=k+3
    plt.show()
        
    
    
#plt.figure(figsize=(12,12))
#n=3
#plt.subplot(n,1,1)
#plt.plot(wvdx,'b')
#plt.subplot(n,1,2)
#plt.plot(wvd_lf,'go')
#plt.plot(wvd_pw,'r*')
#plt.subplot(n,1,3)
#plt.plot(spec,'go')
#plt.plot(loc,val,'r*')
#plt.subplot(n,1,3)
#plt.plot(fftx,'ro')
#plt.subplot(n,1,4)
#plt.plot(fftlog,'go')
#plt.subplot(n,1,5)
#plt.plot(ifftx,'yo')
#plt.subplot(n,1,6)
#plt.plot(pst,'ko')
#plt.show()











