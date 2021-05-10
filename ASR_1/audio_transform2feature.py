# -*- coding: utf-8 -*-
"""
Created on Sun May  9 12:35:51 2021

@author: yxzha

深度学习实际使用的是此module转换的音频特征
"""


import os
import numpy as npy
import pandas as pds
from scipy.signal import lfilter

import re
from matplotlib import pyplot as plt

import  digital_waveform  as dgwv

npy.set_printoptions(suppress=True) # 不用 科学计数法表示


##截断波并返回有效的截断区间
def recg_trunc(wvd,winsize,a,nframes,b=0.0,c=2):
    '''
    识别分割点，并返回有效的音频区间：
    （用frames-point标识,可直接用于wvd索引到点的位置， 或除以framerate就可以转换为所在的时间段用来绘图)
    b # 离散值的程度系数，定义离散阈值
    c # currframe(当前音频点)前后参考的window数
    apply的运算稍微有点费时间
    '''
    ##判断每个音频点是否可截断():
    outlierx=npy.mean(npy.abs(wvd))+npy.std(npy.abs(wvd))*b  #根据音频值的离散情况，判断点的有效性（可保留）
    keep_gtavg=wvd.abs()>outlierx 
    x_indx=pds.Series(range(len(keep_gtavg)))
    tranc_frame_bywins=x_indx.apply(lambda x : 
                                    keep_gtavg[x-int(winsize*c) if x-int(winsize*c)>0  else 0 : x+int(winsize*c)].sum() 
                                    )==0  # curr音频点的前后c个windows中，不存在可保留的音频点，curr点则标记为可截断（即局部无效点），反之不可截断（局部有效点）
    ##提取音频分割点，方便划分音频区间
    tranc_frame_bywins_tem=pds.concat([tranc_frame_bywins[:-1],tranc_frame_bywins[1:].reset_index(drop=True)],axis=1) #错位组合，方便比较
    tranc_point_index=tranc_frame_bywins_tem[tranc_frame_bywins_tem.apply(lambda x :x[0]==x[1],axis=1)==False].index.values #比较 ，确定分割点的位置
    
    ## 判断并保留有用的音频区间 ：
    ###音频区间里面都是局部有效音频点 且 区间内的音频点数大于一个阈值（太大导致有用区间丢失，太小会返回一些无用的区间）
    sec_minpnum=winsize/a*0.8 # 0.8个字的间隔
    point_sec=pds.DataFrame([npy.insert(tranc_point_index,0,0),
                                   npy.concatenate((tranc_point_index,[nframes-1]))
                                   ],index=['start_ps','end_ps']).T   #columns:['start_ps','end_ps']
    point_sec['sec_pnum']=point_sec['end_ps']-point_sec['start_ps']
    point_sec['sec_useless']=tranc_frame_bywins[npy.concatenate((tranc_point_index,[nframes-1]))].reset_index(drop=True)
    effect_sec=point_sec.loc[(point_sec.sec_useless==False) & (point_sec.sec_pnum>=sec_minpnum),['start_ps','end_ps','sec_pnum']].reset_index(drop=True)
    return tranc_frame_bywins,tranc_point_index,effect_sec

##移动平均
def meanshift_wvd(wvd,tm,winsize,winstep,nframes):
    '''
    移动平均方便观察波形，同时为后续切割提供依据
    '''
    win_ave=[]
    curr_win=npy.arange(0,winsize)
    while max(curr_win)<=nframes:
        x_tm=tm[curr_win].mean()
        y_wvd=npy.abs(wvd[curr_win]).mean()+npy.abs(wvd[curr_win]).std()
        win_ave.append((x_tm,y_wvd,curr_win))
        curr_win=curr_win+winstep
    return pds.DataFrame(win_ave,columns=['tm','wvd_mean','curr_win_index'])

def conv2_frames_matrix(fb,winsize,winstep=None):
    '''
    移动窗法切割fb为多帧，生成帧矩阵A
    A属于R^(n,m) : n=ceil(nframes/winstep) ;m=winsize
    
    '''
    if winstep==None:
        winstep=winsize*0.5
    curr_win_start=0
    k=1  
    while curr_win_start<fb.shape[0]:
        fx=fb[curr_win_start:curr_win_start+winsize] # a tem frame 
        if len(fx)<winsize:
            fx=npy.append(fx,npy.array([fx[-1]]*(winsize-len(fx)))) #填充维度不够的帧:末尾值padding
        if k==1:
            fmtx=fx.reshape(1,-1) # # frames_matrix 包含帧的矩阵，每一行为一帧，每个帧包含winsize个音频点 
        else:
            fmtx=npy.concatenate((fmtx,fx.reshape(1,-1)),axis=0)  # 已存在，则append 追加到frames-matrix
        k=k+1
        curr_win_start=curr_win_start+winstep
    return fmtx

def waveform_cut2framesfeature(wave_data,framerate,nframes,is_normaled=False):
    #print('1 处理')
    if is_normaled==False:
        wave_data=wave_data-npy.mean(wave_data) # 去中心化
        wvd=wave_data * 1.0 / (max(abs(wave_data))) #归一
        wvd=lfilter([1, -0.99], [1],wvd) # 预加重
        wvd=pds.Series(wvd*1.0/max(abs(wvd))) #再归一
        
        #print('2 有效语音块截取')
        # 1. 截断长语音，提取出有效的语音块 effect_sec
        nwd_1s=10 # 截断时使用，越大切割的越锋利，但不建议大于10
        a=0.1 #移动平均窗口大小的缩放系数（占一个单词的frames数量（音频长度）的比例）
        winsize=int(framerate/nwd_1s*a)  # 窗口大小,这里假设为一个字的音频长度的1/5，对应一个音素的长度
        winstep=int(winsize*0.5) # 窗口每次移动的步长
    
        #  截断波并返回有用的截断区间：
        ### 判断点的可截断性，识别其中的边缘截点（分割点），判断有用的音频区间
        tranc_frame_bywins,tranc_point_index,effect_sec=recg_trunc(wvd,winsize,a,nframes,b=0.5,c=2) 
        ### 将有效区间effect_sec合并
        fb=npy.array([]) # frames-block that useful 
        for rowx in effect_sec.values:
            block_start,block_end,block_pnum=rowx
            fb=npy.append(fb,wvd[block_start:block_end].values) # 将有效区间合并
    else:
        fb=wave_data
    #print('3 移动窗划分帧区间')
    nwd_1s=8
    a=1/3. #移动平均窗口大小的缩放系数（占一个单词的frames数量（音频长度）的比例）
    winsize=int(framerate/nwd_1s*a)  # 窗口大小,这里假设为一个字的音频长度的1/3，对应一个音素的长度
    winstep=int(winsize*0.5) # 窗口每次移动的步长
    
    ## 2 使用移动窗 划分有效音频块frame-block 为 帧的集合（矩阵：M×N => fn × winsize）
    fn=npy.ceil(fb.shape[0]/winstep) #frames_num 
    fmtx=conv2_frames_matrix(fb,winsize=winsize,winstep=winstep)
    return fmtx,winsize
    
if __name__=='__main__':
    # main 中调试，可直接运行
    file_path='E:/algorithm_data/speaches_wav_type/'
    fl_names=os.listdir(file_path)
    flx='你好，你好，你好，你好，你好，你好.wav' 
    speech_text=flx.split('.')[0] #话
    speech_text=re.sub(r'[^\u4e00-\u9fa5]','',speech_text)
    
    # 读取
    wave_data,nframes,nchannels,framerate= dgwv.get_wav_frame(file_name=file_path+flx)
    
    '''以下被封装到了 waveform_cut2framesfeature()方法中 供外部mudule调用'''
    # 预处理
    print('1 处理')
    wave_data=wave_data-npy.mean(wave_data) # 去中心化
    wvd=wave_data * 1.0 / (max(abs(wave_data))) #归一
    wvd=lfilter([1, -0.99], [1],wvd) # 预加重
    wvd=pds.Series(wvd*1.0/max(abs(wvd))) 
    
    print('2 有效语音块截取')
    # 1. 截断长语音，提取出有效的语音块 effect_sec
    nwd_1s=10 # 截断时使用，越大切割的越锋利，但不建议大于10
    a=0.1 #移动平均窗口大小的缩放系数（占一个单词的frames数量（音频长度）的比例）
    winsize=int(framerate/nwd_1s*a)  # 窗口大小,这里假设为一个字的音频长度的1/5，对应一个音素的长度
    winstep=int(winsize*0.5) # 窗口每次移动的步长

    # 截断波并返回有用的截断区间：
    ### 判断点的可截断性，识别其中的边缘截点（分割点），判断有用的音频区间
    tranc_frame_bywins,tranc_point_index,effect_sec=recg_trunc(wvd,winsize,a,nframes,b=0.5,c=2) 
    ### 将有效区间effect_sec合并
    fb=npy.array([]) # frames-block that useful 
    for rowx in effect_sec.values:
        block_start,block_end,block_pnum=rowx
        fb=npy.append(fb,wvd[block_start:block_end].values) # 将有效区间合并
        
    print('3 移动窗划分帧区间')
    # 2 判断元音和辅音帧区间，并表达为字
    nwd_total=len(speech_text)
    ### 值越大，拟合的越准，但是噪音波峰也会更多；值过小，拟合粗糙，会漏掉有用信息；最好的办法是 nwd_1s根据语速自适应
    nwd_1s=8
    a=1/3. #移动平均窗口大小的缩放系数（占一个单词的frames数量（音频长度）的比例）
    winsize=int(framerate/nwd_1s*a)  # 窗口大小,这里假设为一个字的音频长度的1/3，对应一个音素的长度
    winstep=int(winsize*0.5) # 窗口每次移动的步长
    
    ## 使用移动窗 划分有效音频块frame-block 为 帧的集合（矩阵：M×N => fn × winsize）
    fn=npy.ceil(fb.shape[0]/winstep) #frames_num 
    fmtx=conv2_frames_matrix(fb,winsize=winsize,winstep=winstep)
    
    
    
    