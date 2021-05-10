# -*- coding: utf-8 -*-
"""
Created on Sun May  9 12:36:57 2021

@author: yxzha

深度学习网络

务必保证file_path下的.wav文件的framerate一致，即sampling-rate
"""
import os
import re 
import numpy as npy
import pandas as pds

from scipy.signal import lfilter

#import  digital_waveform as dgwf # get_wav_frame
import  digital_waveform  as dgwv
import audio_transform2feature as adtf
import audio_analog_data as adad

from tensorflow.keras.layers import LSTM,Input,Dense
from tensorflow.keras.models import Model

def data_input_preprocess(datas,K1):
    # 预处理
    start_token,stop_token='\t','\n'
    y_wds=set()
    T1=0
    T2=0
    ydata=[]
    xdata=[]
    for x_seq,y_text in datas:
        y_text=[start_token]+list(y_text)+[stop_token]
        y_wds=y_wds.union(set(y_text))
        T1=max(T1,len(x_seq))
        T2=max(T2,len(y_text))
        xdata.append(x_seq)
        ydata.append(y_text)
    
    dicty=dict(pds.DataFrame(sorted(y_wds),columns=['tk']).reset_index()[['tk','index']].values.tolist())
    dicty_revrs=dict(pds.DataFrame(sorted(y_wds),columns=['tk']).reset_index()[['index','tk']].values.tolist())
    #
    N=len(datas)
    K1=K1
    K2=len(dicty)
    
    def mean_shift(x,K1):
        ksub=len(x)-K1
        while ksub>0:
            x=(x[:-1]+x[1:])/2.0
            ksub=len(x)-K1
        return x
    
    encoder_input_data=npy.zeros(shape=(N,T1,K1),dtype='float32')
    decoder_input_data=npy.zeros((N,T2,K2),dtype='float32')
    decoder_output_data=npy.zeros((N,T2,K2),dtype='float32')
    for n,(x,y) in enumerate(zip(xdata,ydata)):
        x=npy.apply_along_axis(lambda xx: mean_shift(x=xx,K1=K1),1,x)
        encoder_input_data[n,0:len(x),:]=x
        for t2,wdy in enumerate(y):
            decoder_input_data[n,t2,dicty[wdy]]=1.0
            if t2>0:
                decoder_output_data[n,t2-1,dicty[wdy]]=1.0
    params={'N':N,
            'T1':T1,
            'T2':T2,
            'K1':K1,
            'K2':K2}
    token_dicts={'y': dicty,
                 'y_revrs': dicty_revrs}
    fit_data={'encoder_input_data':encoder_input_data,
              'decoder_input_data':decoder_input_data,
              'decoder_output_data':decoder_output_data}
    
    return fit_data, params, token_dicts

if __name__=='__main__':
    # 1 读取原始数据
    file_path='E:/algorithm_data/speaches_wav_type/'
    file_names=os.listdir(file_path) ;  ord_f=[2, 6, 8, 9, 10, 11, 5, 7, 1, 0, 3, 4]
    file_names=[ file_names[ord_f.index(i)] for i in range(len(file_names))]
    total_num=len(file_names)
    
    # 2 音频特征化
    datas=[];
    K1=10**6
    for sentencey in file_names[]:
        #sentencey='你好，你好，你好，你好，你好，你好.wav' 
        speech_text=sentencey.split('.')[0] #话
        speech_text=re.sub(r'[^\u4e00-\u9fa5]','',speech_text)
        # 读取音频
        wave_data,nframes,nchannels,framerate= dgwv.get_wav_frame(file_name=file_path+sentencey)
        # 音频特征转化
        feature_seq,K=adtf.waveform_cut2framesfeature(wave_data,framerate,nframes)
        print(speech_text,K)
        K1=min(K1,K)
        datas.append((feature_seq,speech_text))
    
    # 3 算法数据和参数准备
    fit_data, params, token_dicts=data_input_preprocess(datas=datas,K1=K1)
    encoder_input_data=fit_data['encoder_input_data']
    decoder_input_data=fit_data['decoder_input_data']
    decoder_output_data=fit_data['decoder_output_data']
    N,T1,T2,K1,K2=params['N'],params['T1'],params['T2'],params['K1'],params['K2']
    dicty,dicty_revrs=token_dicts['y'],token_dicts['y_revrs']
    
    # 4 计算图搭建
    # 1) 编码器定义 encoder  define
    num_neurons=128
    encoder_inputs=Input(shape=(None,K1)) 
    encoder=LSTM(num_neurons,return_state=True) # 返回中间结果
    encoder_outputs,state_h,state_c=encoder(encoder_inputs) 
    encoder_states=(state_h,state_c)  # 思想向量
    
    # 2) 解码器定义 decoder define
    decoder_inputs=Input(shape=(None,K2))
    decoder_lstm=LSTM(num_neurons,return_sequences=True,return_state=True) #返回序列结果
    decoder_outputs,_,_=decoder_lstm(decoder_inputs,initial_state=encoder_states ) # 使用encoder的state结果
    decoder_dense=Dense(K2,activation='softmax')
    decoder_outputs=decoder_dense(decoder_outputs)
    
    # 3) 组合encoder-decoder model
    mdl=Model(inputs=[encoder_inputs,decoder_inputs],
              outputs=decoder_outputs,
              name='audio2sentence_zhangyaxu') # 此时网络图已搭建好
    
    # 4) compile seq2seq model and fit
    '''
    #categorical_crossentropy:多分类问题使用交叉熵损失函数
    #rmsprop : root mean square prop :是adaGrad的改进，
               区别在于rmsprop在计算梯度时不是暴力累加平方梯度，而是分配了权重p来控制历史梯度的影响 r->p*r+(1-p)g^2
    '''
    mdl.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['acc']) 
    #
    batch_size=64
    epochs=50
    mdl.fit([encoder_input_data,decoder_input_data],
            decoder_output_data,
            batch_size=batch_size,
            epochs=epochs)
    
    # 5  生成序列：  需要自己重构训练层的结构，然后进行重新组装用于predict序列
    #51 编码器模型
    encoder_mdl=Model(inputs=encoder_inputs,outputs=encoder_states) 
    #52 解码器模型
    thought_input=[Input(shape=(num_neurons,)),
                   Input(shape=(num_neurons,))]
    decoder_outputs,state_h,state_c=decoder_lstm(decoder_inputs,initial_state=thought_input)
    decoder_states=[state_h,state_c]
    decoder_outputs=decoder_dense(decoder_outputs)
    decoder_mdl=Model(inputs=[decoder_inputs]+thought_input,
                      outputs=[decoder_outputs]+decoder_states)
    
    #6 预测  生成序列
    # 61 构建生成函数
    #encoder_mdl,decoder_mdl,T2,K2,reverse_dict_ytoken_index,start_token='\t',stop_token='\n'
    start_token='\t';stop_token='\n'
    def decode_seq(encoder_input_seq):
        target_seq=npy.zeros(shape=(1,1,K2))
        target_seq[0,0,dicty[start_token]]=1. # 不应该传入 stop_token
        generate_seq=[]
        thought=encoder_mdl.predict(encoder_input_seq)
        stop_condition=False
        while not stop_condition:
            output_ytoken,h,c=decoder_mdl.predict([target_seq]+thought)
            output_ytoken_idx=npy.argmax(output_ytoken[0,-1,:])
            generate_char=dicty_revrs[output_ytoken_idx]
            #print(generate_char)
            generate_seq=generate_seq+[generate_char]
            if (generate_char==stop_token or len(generate_seq)>T2):
                stop_condition=True
            target_seq=npy.zeros(shape=(1,1,K2))
            target_seq[0,0,output_ytoken_idx]=1.0
            thought=[h,c]
        return generate_seq
    
    # 62 
    def seq_predict(x_seq):
        encoder_input_seq=npy.zeros((1,T1,K1),dtype='float32')
        encoder_input_seq[0,0:len(x_seq),:]=x_seq[0:T1]
        decoder_sentence=decode_seq(encoder_input_seq)
        return decoder_sentence
    
    ## 模拟一份数据 （模拟数据不好用的，还是得重新生成，如果数据量足够，建议自己准备train-data 和 test-data）
    dictx=adad.generate_audio_dict(file_path) #建议数据集不变的情况下保存，因为耗时间。
    acc_num=0;NT=1000
    error_sample=[]
    for i in range(NT):
        x_test,y_test=adad.analog_wave_data(dictx)
        feature_seq,K=adtf.waveform_cut2framesfeature(wave_data=x_test,framerate=framerate,nframes=len(x_test),is_normaled=True)
        y_pred=seq_predict(x_seq=feature_seq)
        y_pred.remove('\n')
        if y_test==y_pred:
            acc_num+=1
        else:
            error_sample.append([y_test,y_pred])
            pass #print(y_test ,y_pred)
    print('预测序列的完全准确率 {0} % : '.format(acc_num/NT*100))   

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
