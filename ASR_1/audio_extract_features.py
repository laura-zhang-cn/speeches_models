# -*- coding: utf-8 -*-
"""
Created on Mon May  3 09:45:28 2021

@author: yxzha

特征是共振峰，这里仅提供了提取共振峰特征的方法，实际训练时并未使用此特征。

"""

import os
import re 
import numpy as npy
import pandas as pds

from scipy.signal import lfilter

#import  digital_waveform as dgwf # get_wav_frame
import waveform_cut as wvct
import formant_feature as fmft

from matplotlib import pyplot as plt

if __name__=='__main__':
    # 读取数据
    file_path='E:/algorithm_data/speaches_wav_type/'
    file_names=os.listdir(file_path) ;  ord_f=[2, 6, 8, 9, 10, 11, 5, 7, 1, 0, 3, 4]
    file_names=[ file_names[ord_f.index(i)] for i in range(len(file_names))]
    total_num=len(file_names)
    max_formance_loc=0
    for ix,flx in enumerate(file_names[7:]):
        #flx=file_names[1]
        speech_text=flx.split('.')[0] #话
        speech_text=re.sub(r'[^\u4e00-\u9fa5]','',speech_text)
        
        # 字的帧向量
        rst=wvct.waveform_cut2wordphonemesdframes(file_name=file_path+flx,speech_text=speech_text,return_plt_need=True) 
        wphf,tranc_frame_bywins,tm,wvd,win_ave,tranc_point_index,framerate,fb,winstep,wd_pos,phonemes_types,fmtx_zrld_normal,fn=rst
        '''
        # 绘图观察 截断效果和拟合效果
        wvd_trc=wvd[tranc_frame_bywins] # 无效音频区
        tm_trc=tm[tranc_frame_bywins] # 无效音频区
        plt.figure(figsize=(16,5))
        plt.plot(tm,wvd,'go',label='ori-wave') # 原始波
        plt.plot(tm_trc,wvd_trc,'b*',label='tranc-wave') # 截断波区间
        plt.plot(win_ave.tm,win_ave.wvd_mean,'r--*',label='fit-wave') #拟合波
        for x in tranc_point_index:
            plt.plot([x/framerate,x/framerate],[-0.5,0.5],'k--') # 添加截断线
        plt.legend(shadow=True, fontsize='x-large')
        plt.title('波形截断 b {0},c {1} ;移动平均拟合\nspeech:"{2}"'.format(0.5,2,speech_text),fontsize=14)
        plt.grid()
        plt.show()
        
        
        #绘图观察 1-zr ,ld，确定元音辅音的分割帧,对应到字展示
        plt.figure(figsize=(16,10))
        plt.subplot(2,2,(1,2))
        x=(npy.arange(0,len(fb))-winstep)/winstep
        plt.plot(x,fb,'y')
        k=0
        for tx,ps in wd_pos:
            plt.text(ps,0,tx+'\n'+str(phonemes_types[k]))
            k+=1
        yavg,ystd=npy.mean(fmtx_zrld_normal[:,0]),npy.std(fmtx_zrld_normal[:,0])
        plt.plot(fmtx_zrld_normal[:,0],'bo-')
        plt.plot([0,fn+1],[yavg,yavg],'r-')
        plt.plot([0,fn+1],[yavg-ystd,yavg-ystd],'r--')
        plt.plot([0,fn+1],[yavg+ystd,yavg+ystd],'r--')
        plt.grid()
        plt.title(speech_text+': 1-zr')
        #
        plt.subplot(2,2,(3,4)) 
        x=(npy.arange(0,len(fb))-winstep)/winstep
        plt.plot(x,fb,'y')
        k=0
        for tx,ps in wd_pos:
            plt.text(ps,0,tx+'\n'+str(phonemes_types[k]))
            k+=1
        yavg,ystd=npy.mean(fmtx_zrld_normal[:,1]),npy.std(fmtx_zrld_normal[:,1])
        plt.plot(fmtx_zrld_normal[:,1],'bo-')
        plt.plot([0,fn+1],[yavg,yavg],'r-')
        plt.plot([0,fn+1],[yavg-ystd,yavg-ystd],'r--')
        plt.plot([0,fn+1],[yavg+ystd,yavg+ystd],'r--')
        plt.grid()
        plt.title(speech_text+': ld')
        plt.show()
        '''
        
        normal_len=npy.ceil(framerate/4/3/1000)*1000 # 由于语速快慢，使得每个音素所占的窗口帧的长度不同，提取的共振峰位置就没有可比性，所以需要标准化为共同的长度
        # 提取共振峰特征
        def formant_cep(frame_vec,winsize=10):
            wvd_lf=lfilter([1, -0.99], [1],frame_vec) # 传入帧：大概代表一个音素的音频点长度,预加重
            spec,val,loc=fmft.cepstrum_formant(wvd_lf,winsize=winsize)
            return  loc[0:2]
        
        wcpfm=[]
        max_formance_loc=0
        for wdx,wvc in wphf:
            vecx=npy.apply_along_axis(lambda x: formant_cep(x)[0],1,wvc) # 只用第一峰
            max_formance_loc=max(max_formance_loc,max(vecx))
            wcpfm.append((ix,wdx,vecx))
            
        #   
        max_floc=[]
        if ix ==0:
            rst=pds.DataFrame(wcpfm,columns=['idx','word','formant_loc'])
            max_floc.append(max_formance_loc)
        else:
            rst=pds.concat((rst,pds.DataFrame(wcpfm,columns=['idx','word','formant_loc'])),axis=0,ignore_index=True)
        
    
    
    # rst即为处理完成的语音的共振峰特征
        
        
        
        
        
        
        

        
        
        
        
        
        
        
        
        