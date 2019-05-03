# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 10:01:50 2019

@author: a-kojima
"""
import numpy as np
import numpy.matlib as npm
import soundfile as sf
from deep_waveform_dnn import deep_waveform_dnn
from tensorflow.keras import backend as K
from tensorflow.keras import models
import glob
from sklearn.preprocessing import LabelBinarizer
import sys

# ==================
# analysis parameters
# ==================
SAMPLING_FREQUENCY = 16000
AUDIO_SAMPLE_DUR = 0.275
NUMBER_OF_CHANNELS = 2
IMPULSE_RESPONSE_LENGTH = 400
NUMBER_OF_FILTERS = 40 # band-pss filter
FRAME_DUR = 0.025
FRAME_SHIFT_DUR = 0.01
NUMBER_OF_SAMPLE = 10000
DEBUG = False
GAMMATONE_WEIGHT_PATH = r'./gammatone_weight.npy' # for gamma-tone init


number_of_relu_layer = 1
number_of_relu_node = 320

class opt:
    def zero_mean_unit_variance(sample):
        return (sample - np.mean(sample)) / np.std(sample)    
    def convert_one_hot_vector(label_sequence):
        trans = LabelBinarizer()
        one_hot_vector = trans.fit_transform(label_sequence)
        return one_hot_vector
        





audio_length = np.int(AUDIO_SAMPLE_DUR * SAMPLING_FREQUENCY)
'''
model_list = np.load(r'./model_list.npy')

data = np.zeros((1, audio_length, NUMBER_OF_CHANNELS), dtype=np.float32)
label = np.array([], dtype=np.str)



for model in model_list:
    if DEBUG:
        wavform_name_list = glob.glob('./' + str(model) + '/' + '**.wav')[0:30]
    else:
        wavform_name_list = glob.glob('./' + str(model) + '/' + '**.wav')
    #print(len(wavform_name_list))
    for wavform_name in wavform_name_list:
        print(wavform_name)
        wavform, _ = sf.read(wavform_name, dtype='float32')
        wavform = opt.zero_mean_unit_variance(wavform)
        data = np.concatenate((data, npm.reshape(wavform, [1, audio_length, NUMBER_OF_CHANNELS])))
        label = np.append(label, str(model))
        
data = data[1:, :, :]
one_hot_vector = opt.convert_one_hot_vector(label)

print('label', np.shape(one_hot_vector))
print('data', np.shape(data))

np.save('label.npy', label)
np.save('data.npy', data)
'''
label = np.load('./label.npy')
data = np.load('./data.npy')
one_hot_vector = opt.convert_one_hot_vector(label)

K.set_learning_phase(1) 
K.clear_session()
deep_waveform_dnn = deep_waveform_dnn(NUMBER_OF_FILTERS,
                                      NUMBER_OF_CHANNELS,
                                      audio_length,
                                      IMPULSE_RESPONSE_LENGTH,
                                      GAMMATONE_WEIGHT_PATH,
                                      sampling_frequency=SAMPLING_FREQUENCY,
                                      number_of_class=len(list(set(label))))

dwd = deep_waveform_dnn.get_model(number_of_dense_layer=number_of_relu_layer,
                                  number_of_ff_node=number_of_relu_node,
                                  optimizer='sgd',
                                  learning_rate=0.01)
dwd.load_weights(r'./deep_waveform_dnn_weight.hdf5')

"""
dwd = deep_waveform_dnn.gammatone_init(dwd)

print(dwd.summary())

train_model = deep_waveform_dnn.train(data, one_hot_vector, dwd, batch_size=batch_size, epochs=17, validation_split=0.2)
#models.save_model(train_model, model_name)
train_model.save_weights('./deep_waveform_dnn_weight.hdf5')
"""