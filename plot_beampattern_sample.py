# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 11:44:47 2019

@author: a-kojima
"""
import numpy as np
import matplotlib.pyplot as pl
import soundfile as sf
from scipy import signal as sg
from scipy.fftpack import fft
import copy
    
class PlotBeamPattern:
    def __init__(self, gammatone_path, sampling_frequency=16000):
        self.gammatone_path = gammatone_path
        self.sampling_frequency = sampling_frequency
    def plot_beampattern(self, speech_path, selected_filter_indexes, fftl=2048, frame_length=128, frame_shift=32):
        gammatone_weight = np.load(self.gammatone_path)
        if np.ndim(gammatone_weight) == 4:  
            gammatone_weight = np.transpose(gammatone_weight[:, :, :, 0], (2, 1, 0))
        filter_order, number_of_channels, number_of_filter = np.shape(gammatone_weight)
        PHASESHIFTSIZE = 20                         
        LOOKDIRECTION = 36
        pl.figure(),        
        index_plot = 1        
        for selected_filter_index in selected_filter_indexes:
            filter_response = gammatone_weight[:, :, selected_filter_index]
            all_data, _  = sf.read(speech_path, dtype='float32')    
            time_axis = (np.linspace(0, len(all_data) / self.sampling_frequency, filter_order)) * 1000
            number_of_frames = np.int((len(all_data) - frame_length) / frame_shift)
            st = 0
            ed = frame_length            
            beam_pattern = np.zeros((LOOKDIRECTION, np.int(fftl / 2) + 1)) 
            for ii in range(0, number_of_frames):
                data = all_data[st:ed, :]
                ch1_filter = sg.lfilter(filter_response[:, 0], 1, data[:, 0])
                ch2_filter = sg.lfilter(filter_response[:, 1], 1, data[:, 1])        
                for j in range(0, LOOKDIRECTION):
                    roll_range = PHASESHIFTSIZE * np.int(j - np.int(LOOKDIRECTION / 2))
                    ch2_filter_shift = copy.deepcopy(np.roll(ch2_filter, roll_range ))                        
                    if roll_range >= 0:
                        ch2_filter_shift[0:roll_range ] = 0
                    else:
                        ch2_filter_shift[roll_range:- 1] = 0
                    filter_sum = ch1_filter + ch2_filter_shift        
                    beam_pattern[j, :] = beam_pattern[j, :] + np.abs(fft(filter_sum, n=fftl)[0:np.int(fftl / 2) + 1]) ** 2
                st = st + frame_shift
                ed = ed + frame_shift
            vmin = np.min(beam_pattern)
            vmax = np.max(beam_pattern)
            pl.subplot(len(selected_filter_indexes), 2, index_plot)
            pl.plot(time_axis, filter_response)
            if index_plot==1:
                pl.title('Filter Coeficient')
            if index_plot == len(selected_filter_indexes) -1:
                pl.xlabel('Time[ms]')    
            pl.subplot(len(selected_filter_indexes), 2, index_plot + 1)
            pl.imshow(beam_pattern, aspect='auto', origin='lower', cmap='hot', extent = [0, 8, 0, 180], vmin=vmin, vmax=vmax, interpolation='nearest')
            if index_plot + 1 == 2:
                pl.title('Beampattern')
            if index_plot == len(selected_filter_indexes):
                pl.xlabel('Frequency[kHz]')    
            pl.ylabel('DOA [deg]')        
            pl.ylim([0, 180])
            pl.xlim([0, 8])
            index_plot = index_plot + 2        
            beam_pattern = beam_pattern * 0
        
    def plot_brainogram(self, speech_path, frame_length=1024, frame_shift=256):            
        gammatone_weight = np.load(self.gammatone_path)
        if np.ndim(gammatone_weight) == 4:  
            gammatone_weight = np.transpose(gammatone_weight[:, :, :, 0], (2, 1, 0))
        index = self._get_sort_index(gammatone_weight)
        gammatone_weight = gammatone_weight[:, :, index]
        filter_order, number_of_channels, number_of_filter = np.shape(gammatone_weight)            
        all_data, _  = sf.read(SAMPLE_SPEECH, dtype='float32')
        number_of_frames = np.int((len(all_data) - frame_length) / frame_shift)
        brainogram = np.zeros((number_of_filter, number_of_frames, number_of_channels))
        pl.figure()
        for i in range(0, number_of_channels):
            for j in range(0, number_of_filter):
                filter_response = gammatone_weight[:, :, j]
                all_data, _  = sf.read(SAMPLE_SPEECH, dtype='float32')                
                number_of_frames = np.int((len(all_data) - frame_length) / frame_shift)
                st = 0
                ed = frame_length
                for ii in range(0, number_of_frames):
                    data = all_data[st:ed, :]
                    filter_data = sg.lfilter(filter_response[:, i], 1, data[:, i])
                    brainogram[j, ii, i] = np.sum(filter_data ** 2)                                        
                    st = st + frame_shift
                    ed = ed + frame_shift
            pl.subplot(number_of_channels, 1, i + 1)                
            pl.imshow(10 * np.log10(brainogram[:, :, i] ** 2),aspect='auto', origin='lower', cmap='hot')

    def _get_sort_index(self, model_weight):
        _, _, n_f = np.shape(model_weight)
        imp = np.zeros(1024)
        imp[0] = 1
        pos = np.array([])
        for ii in range(0, n_f):
            test = np.abs(fft(sg.lfilter(model_weight[:, 0, ii], 1, imp)))[0:513]
            ind = np.argmax(test)
            pos = np.append(pos, ind)
        return np.argsort(pos)


if __name__ == '__main__':
    SAMPLING_FREQUENCY = 16000
    SAMPLE_SPEECH = r'./test0128.wav'
    GAMMATONE_WEIGHT = r'./train_gammatone_weight.npy'
    plot = PlotBeamPattern(GAMMATONE_WEIGHT,SAMPLING_FREQUENCY)
    plot.plot_beampattern(SAMPLE_SPEECH, [0, 5, 10, 20])
    plot.plot_brainogram(SAMPLE_SPEECH)
    
