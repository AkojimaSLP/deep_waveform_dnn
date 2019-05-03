# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 17:46:21 2019

@author: a-kojima
"""
import numpy as np
from tensorflow.keras import layers, losses, utils, Model, optimizers
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping

SAMPLING_FREQUENCY = 16000
AUDIO_SAMPLE_DUR = 0.275
NUMBER_OF_CHANNELS = 2
IMPULSE_RESPONSE_LENGTH = 1024
NUMBER_OF_FILTERS = 40 # band-pss filter
FRAME_DUR = 0.025
FRAME_SHIFT_DUR = 0.01
NUMBER_OF_SAMPLE = 10000
NUMBER_OF_CLASS = 3 # = number of phones
GAMMATONE_WEIGHT_PATH = r'./gammatone_weight.npy'

# ==============================================================================
class generate_data:
    def __init__(self,
                 number_of_sample,
                number_of_class,
                audio_sample_length,
                sampling_frequency=16000,                
                ):
        self.number_of_sample = number_of_sample
        self.number_of_class = number_of_class
        self.sampling_frequency = sampling_frequency
        self.audio_sample_length = audio_sample_length
    
    def generate_sample(self):
        return np.random.rand(self.audio_sample_length, self.number_of_sample)
    
    def generate_label(self):
        labels = np.random.randint(0, high=self.number_of_class, size=self.number_of_sample)
        return utils.to_categorical(labels, num_classes=self.number_of_class)
# ==============================================================================
    
class deep_waveform_dnn:    
    
    def __init__(self, 
                 F, # number of banpass filter
                 M, # number of channels                 
                 audio_sample_length,
                 impulse_reponse_length,
                 gammatone_weight_path,
                 sampling_frequency=16000,
                 fft_length=400,
                 fft_shift=160,
                 number_of_class=1000):
        self.F = F
        self.M = M
        self.audio_sample_length = audio_sample_length
        self.impulse_reponse_length = impulse_reponse_length
        self.gammatone_weight_path = gammatone_weight_path
        self.sampling_frequency = sampling_frequency
        self.fft_length = fft_length
        self.fft_shift = fft_shift
        self.number_of_class = number_of_class
    
    def gammatone_init(self, model):
        gammatone_weight = np.load(self.gammatone_weight_path).astype('float32')
        gammatone_weight = np.reshape(gammatone_weight, [self.impulse_reponse_length, 1, self.F])
        bias = model.layers[1].get_weights()[1]
        model.layers[1].set_weights(list([np.repeat(gammatone_weight, self.M, axis=1), bias]))
        return model
    
    def hop_and_maxpooling(self, convolution_output):
        '''
        Input: 
            batch size * speech length * number of filters
        '''
        number_of_frames = np.int((self.audio_sample_length - self.fft_length) / self.fft_shift)
        number_of_sample = K.shape(convolution_output)[0]  
        non_linearity = []

        convolution_output_relu = K.relu(convolution_output)   
        
        for i in range(0, self.F):            
            filtered_x = convolution_output_relu[:, :, i]  
            start_point = 0
            stop_point = self.fft_length            
            for j in range(0, number_of_frames):
                filtered_x_window = filtered_x[:, start_point:stop_point]
                max_value = K.log(K.max(filtered_x_window, axis=1) + 0.01) 
                non_linearity.append(max_value)
                start_point = start_point + self.fft_shift
                stop_point = stop_point + self.fft_shift  
        return K.reshape(K.cast(non_linearity, dtype='float32'), (number_of_sample, self.F * number_of_frames))

    def get_shape_hop_and_maxpooling(self, input_shape):
        number_of_frames = np.int((self.audio_sample_length - self.fft_length) / self.fft_shift)
        return tuple(None, number_of_frames * self.F)
                
    def get_model(self, number_of_dense_layer=4, number_of_ff_node=640, optimizer='sgd', learning_rate=0.01):
        input_sequence = layers.Input(shape=(self.audio_sample_length, self.M))
        conv_layer = layers.Conv1D(self.F, self.impulse_reponse_length, padding='same')(input_sequence)
        hop_and_pooling_layer = layers.Lambda(self.hop_and_maxpooling,
                                              output_shape=self.get_shape_hop_and_maxpooling,
                                              trainable=False)(conv_layer)
        full_connect_layer = layers.Dense(number_of_ff_node, activation='relu')(hop_and_pooling_layer)
        for _ in range(0, number_of_dense_layer):
            full_connect_layer = layers.Dense(number_of_ff_node, activation='relu')(full_connect_layer)
            full_connect_layer = layers.Dense(number_of_ff_node, activation='relu')(full_connect_layer)
            full_connect_layer = layers.Dense(number_of_ff_node, activation='relu')(full_connect_layer)
        full_connect_layer = layers.Dense(self.number_of_class, activation='softmax')(full_connect_layer)
        model = Model(inputs=input_sequence, outputs=full_connect_layer)
        if optimizer == 'sgd':
            model.compile(optimizer=optimizers.SGD(lr=learning_rate), loss=losses.categorical_crossentropy, metrics=['accuracy'])        
        elif optimizer == 'adagrad':
            model.compile(optimizer=optimizers.Adagrad(lr=learning_rate), loss=losses.categorical_crossentropy, metrics=['accuracy'])        
        return model
    
    def train(self, ch_multi, label, model, batch_size=100, epochs=10, validation_split=0.3):
        model.fit(ch_multi, label, batch_size=batch_size, epochs=epochs, validation_split=validation_split, callbacks=[EarlyStopping(monitor='val_loss', verbose=1, patience=5)])
        return model    