import os
import config


spectogram_dir = config.INPUT_DIR + 'CodecAugmented_spectogram/'  

'''
I have written CodecAugmented spectrogram because, these audio files are compressed with OPUS audio codec
with bitrate of 12 kbps,  We will take different bitrates for the audio files and add them into the
original spectograms and then run the model for each case.
'''

if not os.path.exists(spectogram_dir):
    os.makedirs(spectogram_dir)

if not os.path.exists(spectogram_dir +'anger'):
    os.makedirs(spectogram_dir +'anger')
if not os.path.exists(spectogram_dir + 'sadness'):
    os.makedirs(spectogram_dir +'sadness')
if not os.path.exists(spectogram_dir + 'happiness'):
    os.makedirs(spectogram_dir + 'happiness')
if not os.path.exists(spectogram_dir +'neutral'):
    os.makedirs(spectogram_dir + 'neutral')