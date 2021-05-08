# 1. 预处理，生成特征向量：  
> 1) 读取音频的时域数据 , 并 归一化处理   
> 2) 切割音频波:   
> 提取有效音频区间A →   
> 使用移动窗将每个A切割为多个帧的集合F    
  
> 3) 为每个字匹配帧（合并帧区间） ：使用帧的强度预测帧集合F中的帧所属的字，使用双门法将每个字区分元音、辅音帧区间B
> 4) 提取帧区间B的特征（这里使用共振峰特征）  
> 5) 根据第3和第4部，计算每个字的特征向量  

# 2. seq2seq算法 或 逆HMM算法  
略  

# 3.modules简介  
# 1)waveform_cut  
读取一段话的音频，并对音频进行切割，获得有效的音频区间,  
同时使用移动平均法拟合音频波形，  
**切割和拟合效果：**  
![切割时域音频1](https://github.com/laura-zhang-cn/speeches_models/blob/main/ASR_1/asr1images/img3_wavecut.png)  
![文字位置拟合-拟合前](https://github.com/laura-zhang-cn/speeches_models/blob/main/ASR_1/asr1images/img2_word_cut2.png)  
![文字文字拟合-拟合后](https://github.com/laura-zhang-cn/speeches_models/blob/main/ASR_1/asr1images/img2_word_cut3.png)  
# 2)formant_feature  
提取一个帧区间的共振峰特征  
**共振峰特征提取效果**  
![共振峰特征](https://github.com/laura-zhang-cn/speeches_models/blob/main/ASR_1/asr1images/image1_formant_feature.png)  

