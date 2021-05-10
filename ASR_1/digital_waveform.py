# -*- coding: utf-8 -*-
"""
Created on Mon May  3 10:46:22 2021

@author: yxzha

1）数据准备(readframe)
读取语音，并转换为数字信号供后续算法使用
2）数据处理（preprocess)

"""

# 读Wave文件并且绘制波形
import wave
from  matplotlib import pyplot as plt
import numpy as npy
import pandas as pds

import os

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示符号

def sign_num(x):
    '''
    array x中的值的正负符号，0 用+1代替
    '''
    y=x/npy.abs(x)
    y[npy.isnan(y)]=1.0
    return y

def get_wav_frame(file_name):
    # 打开WAV音频
    f = wave.open(file_name, "rb")
    
    # 读取格式信息
    # (声道数、量化位数、采样频率、采样点数、压缩类型、压缩类型的描述)
    # (nchannels, sampwidth, framerate, nframes, comptype, compname)
    params = f.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    # nchannels通道数 = 2
    # sampwidth量化位数 = 2
    # framerate采样频率即sampling_rate
    # nframes采样点数 
    # 读取nframes个数据，返回字符串格式
    str_data = f.readframes(nframes)
    f.close()
    # 将字符串转换为数组，得到一维的short类型的数组
    wave_data = npy.fromstring(str_data, dtype=npy.short)
    return wave_data,nframes,nchannels,framerate


def plot_waveform1(wavedata_s):
    # 这里默认数据是channel =1时
    wave_num=len(wavedata_s)
    if wave_num>1:
        plt.figure(figsize=(12,npy.ceil(wave_num/2.0)*4))
        #波形
        i=0
        for wavex,tm,textx in wavedata_s:
            print(textx)
            i=i+1
            plt.subplot(npy.ceil(wave_num/2.0), 2, i)
            plt.plot(tm, wavex)
            plt.title(textx,fontsize=14)
            plt.xlabel("时间/s",fontsize=14)
            plt.ylabel("幅度",fontsize=14)
            plt.grid() # 紧密布局
        plt.tight_layout() 
    else:
        wavex,tm,textx=wavedata_s[0]
        plt.figure(figsize=(6,4))
        plt.plot(tm, wavex)
        plt.xlabel("时间/s",fontsize=14)
        plt.ylabel("幅度",fontsize=14)
        plt.title(textx,fontsize=14)
        plt.grid()  # 标尺
    plt.show()

def wave_preprocess(wave_data):
    '''
    预处理，去掉噪音，静音
    需要借助声学模型提取声学特征，是一个比较复杂的科目，
    这里为了快速把整个ASR算法完成，
    先使用简单的时域/频域特征作为输入特征状态，声学特征提取后面再作为特征状态的改进
    时域特征：
    基音周期：波的共振峰提取及切割
    '''
    pass

if __name__=='__main__':
    # 数据读取：
    file_path='E:/algorithm_data/speaches_wav_type/'
    file_names=os.listdir(file_path) 
    ord_f=[8, 0, 2, 3, 6, 1, 7, 9, 4, 5]
    file_names=[ file_names[ord_f.index(i)] for i in range(len(file_names))]
    #flx=file_names[-1]
    total_num=len(file_names)
    plot_data=[]
    for ix,flx in enumerate(file_names):
        speech_text=flx.split('.')[0] #话
        wave_data,nframes,nchannels,framerate= get_wav_frame(file_name=file_path+flx)
        wave_data = wave_data * 1.0 / (max(abs(wave_data))) #归一
        tm=npy.arange(0,nframes)*1.0/framerate # x轴的采样时间
        plot_data.append((wave_data,tm,speech_text))
        
    plot_waveform1(wavedata_s=plot_data)
    
    
    
