# -*- coding: utf-8 -*-
"""
Created on Mon May  3 09:56:48 2021

@author: yxzha

因为使用的系统自带的录音机功能，存储格式为mpeg4(mp4)，需要转换为wav

使用命令工具ffmpeg ，需要自己下载按照到电脑上：
ffmpeg  :
    -i: 表示输入的音频或视频
    -ac: channel 设置通道3, 默认为1
    -ar: sample rate 设置音频采样率
    -acodec: 使用codec编解码
    -ab: bitrate 设置音频码率
    -vn: 不做视频记录(对于视频类型起作用)
    libmp3lame: 使用libmp3lame编码mp3
    -y: 覆盖输出文件
    -f: 强制转换格式（如果输出文件带了后缀，系统可以自动猜测，这个其实也不用加了）
"""

import os
import re

def mp4_to_wav(m4a_files, wav_new, ar,ac,rep=True):
    """
    m4a 转 wav
    :param m4a_files: .m4a文件路径
    :param wav_new: .wav文件路径
    :param sampling_rate: 采样率
    :return: .wav文件
    """
    command = "D:\\Applications\\ffmpeg-18639\\ffmpeg -i {0} -vn -ac {2} -ar {1} {3} && y".format(m4a_files, ar,ac, wav_new)
    if os.path.exists(wav_new):
        if rep:
            os.remove(wav_new)
            print('命令是：',command)
            os.system(command)
        else:
            pass
    else:
        print('命令是：',command)
        # 执行终端命令
        os.system(command)


if __name__ == '__main__':
    m4a_path = r'C:/Users/yxzha/Documents/录音/'
    wav_path = r'E:/algorithm_data/speaches_wav_type/'
    m4a_files=os.listdir(m4a_path)
    m4a_files=['又涨本事了，一次推两个.m4a','又涨本事了，一次推两个。.m4a',]
    for flx in m4a_files:
        fly=''.join(re.split('\.',flx)[:-1])
        mp4_to_wav(m4a_files=m4a_path+flx, wav_new=wav_path+fly+'.wav', ar=22100,ac=1)

'''
file:///C:/Users/yxzha/Documents/录音/夜来风雨声，花落知多少.m4a
file:///C:/Users/yxzha/Documents/录音/春眠不觉晓，处处闻啼鸟.m4a
file:///C:/Users/yxzha/Documents/录音/春晓，唐，孟浩然.m4a
file:///C:/Users/yxzha/Documents/录音/我可没有扭你的胡子，你不要乱说.m4a

'''