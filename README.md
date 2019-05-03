# deep_waveform_dnn implementation
- SPEECH ACOUSTIC MODELING FROM RAW MULTICHANNEL WAVEFORMS
- http://www.cs.huji.ac.il/~ydidh/waveform.pdf

# train time domain filter coefficient (beamformer) from waveform
- CNN layer's weights are regarded as beamformer

# scripts for training and plot beampattern and brainogram
- because of 2ch microphone, we can see 1 null(around 90 degree) in beampattern


![brainogram](https://user-images.githubusercontent.com/41845296/57135607-f3193f80-6de4-11e9-941f-f9d5b88c18c2.png)
![beam_pattern](https://user-images.githubusercontent.com/41845296/57135609-f3193f80-6de4-11e9-8cb4-2c4b91bf9336.png)
