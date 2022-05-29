4#script from homework
from subprocess import Popen
Popen('sudo sh -c "echo performance >'
 '/sys/devices/system/cpu/cpufreq/policy0/scaling_governor"',
 shell=True).wait()


#import section
import tensorflow as tf
import numpy as np
import argparse
import os
import time
from scipy.io import wavfile
import scipy.signal as signal
import math

#cast function
def foo(l, dtype):
  return list(map(dtype, l))


#list of audio files in path
dir_path = 'yes_no'
l=os.listdir(dir_path)

#set lists
list_time_slow=[]
list_mfccs_slow=[]


#set hyperparameters
frame_length=256
frame_step=128
fft_lenght=frame_length
lower_edge_hertz, upper_edge_hertz, num_mel_bins = 20, 4000, 40


for j,i in enumerate(l):
    if(j%100==0):
        print(f'{j}/{len(l)} for the slow phase')

    #start time 
    start = time.time()

    #reading wav file
    rate, audio = wavfile.read(dir_path+'/'+i)

    #stft
    stft = tf.signal.stft(foo(audio,float), frame_length, frame_step,
        fft_length=frame_length)
    spectrogram = tf.abs(stft)


    #num_spectrogram_bins and  linear_to_mel_weight_metrix can be computed only one time
    if j==0:
        num_spectrogram_bins = stft.shape[-1]
        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(num_mel_bins, num_spectrogram_bins, 16000, lower_edge_hertz, upper_edge_hertz)



    # tensor dot product
    mel_spectrogram = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)
    

    #adding little values to mel_spectrogram in order not to have 0 values in log function
    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)

    
    #compute mfcc values and taking 10 values
    
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
    mfccs_slow = mfccs[..., :10]


    #end time
    end = time.time()

    #computing and appending execution time
    list_time_slow.append(end-start)

    #appending mfccs 
    list_mfccs_slow.append(mfccs_slow)

#compute the average execution time of mfcc_slow
print('Slow mean execution Time: {:.4f}s'.format(np.mean(list_time_slow)))

#MFCC_fast section

#hyperparamers
frame_length=64
frame_step=32
fft_length=64

lower_edge_hertz, upper_edge_hertz, num_mel_bins = 20,2000,10
#defining lists
list_mfccs_fast=[]
list_time_fast=[]

audio = []

for j,i in enumerate(l):
    if j % 100 ==0:
        print(f'{j}/{len(l)} for the fast phase')

    #start time
    start = time.time()
    #reading wav file
    rate, audio = wavfile.read(dir_path+'/'+i)

    #resample
    audio = signal.resample_poly(audio,1,4)
       
    #stft
    stft = tf.signal.stft(tf.cast(audio, tf.dtypes.float32), frame_length, frame_step,
        fft_length=fft_length)

    spectrogram = tf.abs(stft)
    spectrogram=tf.expand_dims(spectrogram,-1)
    spectrogram=tf.image.resize(spectrogram,[16,16])
    spectrogram=tf.squeeze(spectrogram)

    #num_spectrogram_bins and linear_to_weight_matrix can be computed only one time
    if j == 0:
        num_spectrogram_bins = (spectrogram).shape[-1]
        linear_to_mel_weight_matrix = tf.cast(tf.signal.linear_to_mel_weight_matrix(num_mel_bins, num_spectrogram_bins, 4000, lower_edge_hertz, upper_edge_hertz), tf.dtypes.float32)
   

    #fot product
    mel_spectrogram = tf.tensordot(spectrogram,linear_to_mel_weight_matrix, 1)
    
    
    #adding little values to  mel_spectrogram in order to not have 0 values in the log function
    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)
    c=log_mel_spectrogram
    c=tf.expand_dims(c,-1)
    c=tf.image.resize(c,[124,32])
    c=tf.squeeze(c)


    #computing  mfcc vector and taking 10 values
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(c)
    mfccs_fast = mfccs[..., :10]
    

    #end time
    end = time.time()

    #computing single execution time
    list_time_fast.append(end-start)

    #list of mafccs
    list_mfccs_fast.append(mfccs_fast)

#average time execution mfcc_fast
print('Fast mean execution Time: {:.4f}s'.format(np.mean(list_time_fast)))




#set snrs vector
snrs=[]

#computing the snr between mfcc_fast and mfcc_slow  
for  mfccs_slow1, mfccs_fast1 in zip(list_mfccs_slow,list_mfccs_fast):
    snr=20*math.log10(np.linalg.norm(mfccs_slow1)/np.linalg.norm(mfccs_slow1-tf.cast(mfccs_fast1+10**-6,tf.dtypes.float32)))
    snrs.append(snr)

#computing and showing the average snr 
print("SNR : {:.4f}dB".format(np.mean(np.array(snrs))))

