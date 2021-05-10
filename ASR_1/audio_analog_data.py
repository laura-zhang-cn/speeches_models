# -*- coding: utf-8 -*-
"""
Created on Sun May  9 16:17:38 2021

@author: yxzha

务必保证file_path下的所有.wav文件的framerate一致哦！

"""
import os
import re
import numpy as npy

import waveform_cut as wvct


def generate_audio_dict(file_path):
    fl_names=os.listdir(file_path)
    wd_and_vec=[]
    for flx in fl_names:
        speech_text=flx.split('.')[0] #话
        speech_text=re.sub(r'[^\u4e00-\u9fa5]','',speech_text)
        #
        rst=wvct.waveform_cut2wordphonemesdframes(file_name=file_path+flx,speech_text=speech_text,continue_k=3,return_plt_need=True)    
        wphf=rst[0]
        winstep=rst[8]
        for wd,mtx in wphf:
            k=len(mtx)-1;start=1
            for vx in mtx:
                if k and start:
                    wd_phone_vec=vx[0:winstep]
                    start=0
                elif k :
                    wd_phone_vec=npy.concatenate((wd_phone_vec,vx[0:winstep]),axis=0)
                else:
                    wd_phone_vec=npy.concatenate((wd_phone_vec,vx),axis=0)
                k=k-1
            wd_and_vec.append((wd,wd_phone_vec))
    dictx=dict(wd_and_vec) # 没有处理多次出现的文字的phone_vec，后面再改进
    return dictx
    

def analog_wave_data(dictx):
    wdall=list('举头望明月低头思故乡你你你你你你你你你你好你好你好你好你好你好又涨本事了一次推两个又涨本事了一次推两个嗨嗨嗨嗨嗨夜来风雨声花落知多少好好好好好好好床前明月光疑是地上霜我可没有扭你的胡子啊不要乱说春晓唐孟浩然春眠不觉晓处处闻啼鸟举头望明月低头思故乡你你你你你你你你你你好你好你好你好你好你好又涨本事了一次推两个又涨本事了一次推两个嗨嗨嗨嗨嗨夜来风雨声花落知多少好好好好好好好床前明月光疑是地上霜我可没有扭你的胡子啊不要乱说春晓唐孟浩然春眠不觉晓处处闻啼鸟')
    
    t=npy.random.randint(1,6) 
    def selectwd(wdall):
        st=npy.random.randint(len(wdall)-3)
        return ''.join(wdall[st:st+3]) 
    wds=''.join([ selectwd(wdall) for ti in range(t)])
    for i,wdx in enumerate(wds):
        if i==0:
            wave_data=npy.array(dictx[wdx]) # frame-block 
        else:
            wave_data=npy.concatenate((wave_data,dictx[wdx]),axis=0)
    return (wave_data,wds)

if __name__=='__main__':
    # main 中调试，可直接运行
    file_path='E:/algorithm_data/speaches_wav_type/'
    #
    dictx=generate_audio_dict(file_path)
    #
    wave_data,wds=analog_wave_data(dictx)
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    