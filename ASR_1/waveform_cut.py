# -*- coding: utf-8 -*-
"""
Created on Tue May  4 15:05:32 2021

@author: yxzha


"""
import os
import numpy as npy
import pandas as pds
from scipy.signal import lfilter

import re
import operator
from functools import reduce
import pypinyin as ppy
from matplotlib import pyplot as plt

import  digital_waveform  as dgwv

npy.set_printoptions(suppress=True) # 不用 科学计数法表示


def sign_num(x):
    '''
    array x中的值的正负符号，0 用+1代替
    '''
    y=x/npy.abs(x)
    y[npy.isnan(y)]=1.0
    return y

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

# 帧的过零率zr ，帧的响度ld
def zr_ld(x):
    '''
    zr: through-zero-rate : 清音过零率高，清音分元辅；浊音过零率低(不过零率高 1-zr），浊音几乎都是辅音；
    ld: frame-loudness ： 清音时，元音更响亮，浊音不太响亮
    '''
    #x=fmtx[0]
    # 相邻点过零判定，-1代表符号相反即过零
    zr=npy.sum((x[1:]*x[:-1])/npy.abs(x[1:]*x[:-1])==-1) / (len(x)-1.0)
    ld=npy.sum(npy.abs(x))
    return 1-zr,ld


def get_peak_pos(x):
    '''
    帧的响度估计值的获得峰值：可能的元音位
    '''
    lower_thr=x.mean()-x.std()*0.9
    #print(lower_thr)
    peak_pos=[]
    idarr=range(1,len(x)-1) # 掐头去尾
    for idx in idarr:
        lfsub=x[idx]-x[idx-1]
        rtsub=x[idx]-x[idx+1]
        cond1=(lfsub>0)&(rtsub>0)
        cond2=(lfsub>0.01)|(rtsub>0.01)
        cond3=x[idx]>lower_thr
        if  cond1 and cond2 and cond3:
            peak_pos.append(idx)
        #print(idx,x[idx-1],x[idx],x[idx+1],lfsub,rtsub)
    return peak_pos

def waveframes_stab(idx,x):
    '''
    曲线的稳定性由波动率和幅度有关
    x: array 1-d ，x∈ [0,1]
    idx : index of x
    '''
    #检测区间为连续的4个帧 ：检测变化度的区间为连续的4个帧（一帧为一音素，一字一般2个音素，帧之间重复率0.5,所以需要4个连续帧）
    n=4 
    tem=x[max(idx-n+1,0):min(idx+n,len(x)-1)]
    range_tem,avgtem=1.0,1.0 #设置一个极大值，因为数据都是归一化的，1.0够了
    for ix in range(0,n):
        temx=tem[ix:ix+n]
        if len(temx)==4:
            #幅度
            range_tem=min(range_tem,npy.max(temx)-npy.min(temx))
            #波动率：累积斜率
            avgtem=min(avgtem,npy.mean(npy.abs(npy.diff(temx))))
            #
    range_total,avg_total=1.0,1.0
    for iy in range(0,len(x)-n):
        temy=x[iy:iy+n]
        if len(temy)==4:
            #幅度
            range_total=max(range_total,npy.max(temy)-npy.min(temy))
            #波动率：累积斜率
            avg_total=max(avg_total,npy.mean(npy.abs(npy.diff(temy))))
            #
    change_rate1=(range_tem+0.001)/(range_total+0.001)
    change_rate2=avgtem/avg_total
    return round(change_rate1,5),round(change_rate2,5)

def izr_check_peak(peak_pos,fmtx_zrld_normal,nwd_total):
    k=len(peak_pos)-nwd_total
    if k==0:
        return peak_pos
    elif k>0:
        ld_change=[]
        izr_change=[]
        for idx in peak_pos:
            ld_change.append(waveframes_stab(idx=idx,x=fmtx_zrld_normal[:,1]))
            #izr_change.append(wave_stab_cal(idx=idx,x=fmtx_zrld_normal[:,0]))
        ldzr_change=pds.DataFrame(ld_change,columns=['ld_range','ld_fluctuate']) #,'izr_range','izr_fluctuate' 
        ldzr_change.insert(0,'idx',peak_pos)
        ldzr_change.insert(1,'ld',fmtx_zrld_normal[peak_pos,1])
        wavelet_rate=[1.0]+list(ldzr_change.ld.values[1:]/ldzr_change.ld.values[:-1])
        ldzr_change.insert(2,'wavelet',wavelet_rate)
        ldzr_change.insert(3,'izr',fmtx_zrld_normal[peak_pos,0])
        ldzr_change=ldzr_change.sort_values(by='ld_range',ascending=True).reset_index(drop=True)
        #准备阈值：过零率高与浊音（元音）条件 违备 ； 余波率大且峰值小于整体
        ldr_lw_thr=ldzr_change.ld_range.mean() #-ldzr_change.ld_range.std()*0.5
        izr_lw_thr=fmtx_zrld_normal[:,0].mean()-fmtx_zrld_normal[:,0].std()
        ld_lw_thr=ldzr_change.ld.median()
        wavelet_lw_thr=0.5
        #
        cond1=ldzr_change.ld_range<=ldr_lw_thr
        low_change=ldzr_change.loc[cond1,['idx','ld','wavelet','izr']]
        # 
        cond2=low_change.izr<=izr_lw_thr
        idx_pop_izr=low_change.loc[cond2,:].sort_values(by='izr',ascending=True)['idx'].values.tolist()[0:k]
        k=k-len(idx_pop_izr)
        if k>0:
            cond3=(low_change.ld<=ld_lw_thr)&(low_change.wavelet<wavelet_lw_thr)
            idx_pop_wavelet=low_change.loc[cond3,:].sort_values(by='ld',ascending=True)['idx'].values.tolist()[0:k]
            k=k-len(idx_pop_wavelet)
        else:
            idx_pop_wavelet=[]
        low_change_idx=idx_pop_izr+idx_pop_wavelet
        for idx in low_change_idx:
            peak_pos.remove(idx)
        return peak_pos
    else:
        return peak_pos 


def double_gates(fmtx,nwd_total):
    '''
    双门法判断元音和辅音：
    短时过零率zr & 短时响度ld
    
    '''
    # 帧的过零率zr ，帧的响度ld(帧的响度与波形拟合，且调整好windowsize 和step，可以完美的拟合字发音 )
    fmtx_zrld=npy.apply_along_axis(zr_ld,1,fmtx) # 逐行计算 1-zr,ld
    fmtx_zrld_normal=npy.apply_along_axis(lambda x: (x-npy.min(x)+0.001)/(npy.max(x)-npy.min(x)+0.001),0,fmtx_zrld)
    
    # 获取锋所在帧位置 即为元音的位置，单词的目标点
    peak_pos=get_peak_pos(fmtx_zrld_normal[:,1])  
    #[1, 9, 22, 27, 35, 39, 45, 54, 59, 75, 81, 90]
    
    # 识别并补充漏掉的字元音或判断和剔除误识别的字元音 :# 对1-zr 和ld 进行低变化和高变化检测 
    peak_pos=izr_check_peak(peak_pos,fmtx_zrld_normal,nwd_total)
    return peak_pos,fmtx_zrld_normal

def phonemes_frames(fmtx,phonemes_types,peak_pos):
    '''
    fmtx : frames-matrix ，窗口帧的矩阵 
    phonemes_types: 0:辅 , 1:元 
    peak_pos: 锋所在的帧 在fmtx中的索引位置
    '''
    wdnum=len(phonemes_types)
    fn=len(fmtx)
    rst=[]
    for idx in range(0,wdnum):
        phone_num=len(phonemes_types[idx])
        if phone_num>=2:
            consonant_frame=fmtx[max(peak_pos[idx]-2,0):min(peak_pos[idx],fn)] # 2个窗口
            vowel_frame=fmtx[max(peak_pos[idx],0):min(peak_pos[idx]+2,fn)] # 2个窗口
        else:
            consonant_frame=fmtx[max(peak_pos[idx],0):min(peak_pos[idx]+2,fn)]  # 与vowel的相同即可
            vowel_frame=fmtx[max(peak_pos[idx],0):min(peak_pos[idx]+2,fn)] 
        rst.append(npy.concatenate((consonant_frame,vowel_frame),axis=0))
    return rst


def waveform_cut2wordphonemesdframes(file_name,speech_text,return_plt_need=False):
    '''
    本function 是为了外部调用使用,与main下运行的基本一致
    '''
    # 参数准备
    wave_data,nframes,nchannels,framerate= dgwv.get_wav_frame(file_name=file_name)
    
    # 预处理
    print('1 处理')
    tm=pds.Series(npy.arange(0,nframes)*1.0/framerate) # x轴的采样时间  time
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

    # 短时平均幅度拟合波形 ：后面使用的是短时能量拟合，效果一样。
    win_ave=meanshift_wvd(wvd,tm,winsize,winstep,nframes)  # 描绘波形趋势，

    # 截断波并返回有用的截断区间：
    ### 判断点的可截断性，识别其中的边缘截点（分割点），判断有用的音频区间
    tranc_frame_bywins,tranc_point_index,effect_sec=recg_trunc(wvd,winsize,a,nframes,b=0.5,c=2) 
    ### 将有效区间effect_sec合并
    fb=npy.array([]) # frames-block that useful 
    for rowx in effect_sec.values:
        block_start,block_end,block_pnum=rowx
        fb=npy.append(fb,wvd[block_start:block_end].values) # 将有效区间合并
    
    print('3 双门法 识别字所在帧区间')
    # 2 判断元音和辅音帧区间，并表达为字
    nwd_total=len(speech_text)
    ### 值越大，拟合的越准，但是噪音波峰也会更多；值过小，拟合粗糙，会漏掉有用信息；最好的办法是 nwd_1s根据语速自适应
    #nwd_1s=7
    nwd_1s=npy.ceil(nwd_total/(len(fb)/framerate))
    a=1/3. #移动平均窗口大小的缩放系数（占一个单词的frames数量（音频长度）的比例）
    winsize=int(framerate/nwd_1s*a)  # 窗口大小,这里假设为一个字的音频长度的1/3，对应一个音素的长度
    winstep=int(winsize*0.5) # 窗口每次移动的步长
    
    ## 使用移动窗 划分有效音频块frame-block 为 帧的集合（矩阵：M×N => fn × winsize）
    fn=npy.ceil(fb.shape[0]/winstep) #frames_num 
    fmtx=conv2_frames_matrix(fb,winsize=winsize,winstep=winstep)
    
    ## 词发音：元辅音
    consonants=ppy.lazy_pinyin(speech_text,style=3) 
    vowels=ppy.lazy_pinyin(speech_text,style=5)
    phonemes=list(zip(consonants,vowels)) # 发音的音素数，注意：有些辅音音素虽然存在，但是不独立发音哦，此种情况直接当作合并元音 
    #phonemes_flat=[x  for x in list(reduce(operator.add,phonemes)) if x!='']
    #phonemes_num=len(phonemes_flat)
    phonemes_types= [(1,) if '' in x else (0,1) for x in phonemes]
    #phonemes_types= list(reduce(operator.add,[(1,) if '' in x else (0,1) for x in phonemes])) # 0:辅，1:元
    
    ## 双门法获取字的靶点位
    peak_pos,fmtx_zrld_normal=double_gates(fmtx,nwd_total)
    wd_pos=list(zip(speech_text,peak_pos))
    
    print('4 字的时域帧区间 ')
    # 根据元音位置，结合字的发音，切开有效音频区间为子帧区间 字：[辅音,元音]
    # 元辅音顺序 : 先辅后元：[consonants , vowels],有辅必有元，有元未必有辅
    word_phonemes_frames=phonemes_frames(fmtx,phonemes_types,peak_pos)
    wphf=list(zip(speech_text,word_phonemes_frames))  # word_phonemes_frames
    
    print('5 读取音频-》识别有效帧区间-》识别字的时域帧区间 完毕  ')
    if return_plt_need:
        return wphf,tranc_frame_bywins,tm,wvd,win_ave,tranc_point_index,framerate,fb,winstep,wd_pos,phonemes_types,fmtx_zrld_normal,fn
    else:
        return wphf

if __name__=='__main__':
    # main 中调试，可直接运行
    file_path='E:/algorithm_data/speaches_wav_type/'
    fl_names=os.listdir(file_path)
    flx='你好，你好，你好，你好，你好，你好.wav' 
    speech_text=flx.split('.')[0] #话
    speech_text=re.sub(r'[^\u4e00-\u9fa5]','',speech_text)
    
    '''以下被封装到了 waveform_cut2wordphonemesdframes()方法中 供外部mudule调用'''
    # 读取
    wave_data,nframes,nchannels,framerate= dgwv.get_wav_frame(file_name=file_path+flx)
    
    # 预处理
    print('1 处理')
    tm=pds.Series(npy.arange(0,nframes)*1.0/framerate) # x轴的采样时间  time
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

    # 短时平均幅度拟合波形 ：后面使用的是短时能量拟合，效果一样。
    win_ave=meanshift_wvd(wvd,tm,winsize,winstep,nframes)  # 描绘波形趋势，

    # 截断波并返回有用的截断区间：
    ### 判断点的可截断性，识别其中的边缘截点（分割点），判断有用的音频区间
    tranc_frame_bywins,tranc_point_index,effect_sec=recg_trunc(wvd,winsize,a,nframes,b=0.5,c=2) 
    ### 将有效区间effect_sec合并
    fb=npy.array([]) # frames-block that useful 
    for rowx in effect_sec.values:
        block_start,block_end,block_pnum=rowx
        fb=npy.append(fb,wvd[block_start:block_end].values) # 将有效区间合并
    # 绘图观察 截断效果和拟合效果
    '''
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
    
    '''
    
    print('3 双门法 识别字所在帧区间')
    # 2 判断元音和辅音帧区间，并表达为字
    nwd_total=len(speech_text)
    ### 值越大，拟合的越准，但是噪音波峰也会更多；值过小，拟合粗糙，会漏掉有用信息；最好的办法是 nwd_1s根据语速自适应
    #nwd_1s=7
    nwd_1s=npy.ceil(nwd_total/(len(fb)/framerate))
    a=1/3. #移动平均窗口大小的缩放系数（占一个单词的frames数量（音频长度）的比例）
    winsize=int(framerate/nwd_1s*a)  # 窗口大小,这里假设为一个字的音频长度的1/3，对应一个音素的长度
    winstep=int(winsize*0.5) # 窗口每次移动的步长
    
    ## 使用移动窗 划分有效音频块frame-block 为 帧的集合（矩阵：M×N => fn × winsize）
    fn=npy.ceil(fb.shape[0]/winstep) #frames_num 
    fmtx=conv2_frames_matrix(fb,winsize=winsize,winstep=winstep)
    
    ## 词发音：元辅音
    consonants=ppy.lazy_pinyin(speech_text,style=3) 
    vowels=ppy.lazy_pinyin(speech_text,style=5)
    phonemes=list(zip(consonants,vowels)) # 发音的音素数，注意：有些辅音音素虽然存在，但是不独立发音哦，此种情况直接当作合并元音 
    phonemes_flat=[x  for x in list(reduce(operator.add,phonemes)) if x!='']
    phonemes_num=len(phonemes_flat)
    phonemes_types= [(1,) if '' in x else (0,1) for x in phonemes]
    #phonemes_types= list(reduce(operator.add,[(1,) if '' in x else (0,1) for x in phonemes])) # 0:辅，1:元
    
    ## 双门法获取字的靶点位
    peak_pos,fmtx_zrld_normal=double_gates(fmtx,nwd_total)
    wd_pos=list(zip(speech_text,peak_pos))
    
    print('4 字的时域帧区间 ')
    # 根据元音位置，结合字的发音，切开有效音频区间为子帧区间 字：[辅音,元音]
    # 元辅音顺序 : 先辅后元：[consonants , vowels],有辅必有元，有元未必有辅
    word_phonemes_frames=phonemes_frames(fmtx,phonemes_types,peak_pos)
    wphf=list(zip(speech_text,word_phonemes_frames))  # word_phonemes_frames
    
    print('5 读取音频-》识别有效帧区间-》识别字的时域帧区间 完毕  ')
    '''
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
    
 
 
