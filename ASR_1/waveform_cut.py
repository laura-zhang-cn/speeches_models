# -*- coding: utf-8 -*-
"""
@author: yxzha


"""
import numpy as npy
import pandas as pds
import re

from matplotlib import pyplot as plt

import  digital_waveform  as dgwv


def sign_num(x):
    '''
    array x中的值的正负符号，0 用+1代替
    '''
    y=x/npy.abs(x)
    y[npy.isnan(y)]=1.0
    return y

##截断波并返回有用的截断区间
def recg_trunc(wvd,winsize,b=0.0,c=2):
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
def meanshift_wvd(wvd,tm,winsize,winstep):
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

def cut_waveform(wvd,nwd_1s=10,nwd_total=None,):
    
    pass

if __name__=='__main__':
    # 读取
    file_path='E:/algorithm_data/speaches_wav_type/'
    flx='夜来风雨声，花落知多少.wav' # 夜来风雨声，花落知多少
    speech_text=flx.split('.')[0] #话
    wave_data,nframes,nchannels,framerate= dgwv.get_wav_frame(file_name=file_path+flx)
    
    # 预处理
    tm=pds.Series(npy.arange(0,nframes)*1.0/framerate) # x轴的采样时间  time
    wave_data=wave_data-npy.mean(wave_data) # 去中心化
    wvd=pds.Series(wave_data * 1.0 / (max(abs(wave_data)))) #归一
    #wvd_pow=npy.power(wvd_old,2)/100*sign_num(wvd_old) # 预加重，不好用
    
    # 参数准备
    nwd_total=len(re.sub(r'[^\u4e00-\u9fa5]','',speech_text))
    nwd_1s=10
    a=0.1 #移动平均窗口大小的缩放系数（占一个单词的frame数量（音频长度）的比例）
    winsize=int(framerate/nwd_1s*a)  # 窗口大小,这里假设为一个字的音频长度的十分之一
    winstep=int(winsize*0.5) # 窗口每次移动的步长

    # 移动平均拟合波形
    win_ave=meanshift_wvd(wvd,tm,winsize,winstep)  # 描绘波形趋势

    # 截断波并返回有用的截断区间：
    ### 判断点的可截断性，识别其中的边缘截点（分割点），判断有用的音频区间
    tranc_frame_bywins,tranc_point_index,effect_sec=recg_trunc(wvd,winsize,b=0.5,c=2) 
    
    # 按共振峰切割音频，使与文本对应
    
    # 绘图观察 截断效果和拟合效果
    wvd_trc=wvd[tranc_frame_bywins] # 无效音频区
    tm_trc=tm[tranc_frame_bywins] # 无效音频区
    
    plt.figure(figsize=(15,10))
    plt.plot(tm,wvd,'go',label='ori-wave') # 原始波
    plt.plot(tm_trc,wvd_trc,'b*',label='tranc-wave') # 截断波区间
    plt.plot(win_ave.tm,win_ave.wvd_mean,'r--*',label='fit-wave') #拟合波
    for x in tranc_point_index:
        plt.plot([x/framerate,x/framerate],[-1,1.0],'k--') # 添加截断线
    plt.legend(shadow=True, fontsize='x-large')
    plt.title('波形截断 b {0},c {1} ;移动平均拟合\nspeech:"{2}"'.format(0.5,2,speech_text),fontsize=14)
    plt.grid()
    plt.show()
    
    # 密集区可以单独取一个时间段观察
    t=[1.5,3]
    condx=win_ave.tm.between(t[0],t[1])
    plt.figure(figsize=(15,5))
    plt.plot(win_ave.loc[condx,'tm'], win_ave.loc[condx,'wvd_mean'],'r-o')
    plt.title('-'.join([str(x)+'s' for x in t]))
    plt.grid()
    plt.show()
    

    
    
    
    
    
