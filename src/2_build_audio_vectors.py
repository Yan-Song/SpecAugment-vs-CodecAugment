import numpy as np
import pandas as pd 
import librosa
import os, glob, sys
import config
import pickle
from tqdm import tqdm

'''
Note while running this script use "-W ignore" to ignore all the warnings
run this script with

>> Python -W ignore src/2_build_audio_vectors.py

Due to memory constraints it may be not be possible to run all the 5 sessions togather
In that case it is suggested that, you run each session individually. 
So rather than choose range(1,6) >> take [1],[2]...[5] individually
'''

def run(session):
    labels_df = pd.read_csv(config.DF_IEMOCAP + 'df_iemocap_{}.csv'.format(session))
    iemocap_dir = config.IEMOCAP_DIR
    compressed_dir = 'input/Compressed_audio/IEMOCAP_Compressed_br64_fs5/'
    sr = 44100 #sample rate 
    audio_vectors = {}
    
    for sess in [session]:
        wav_file_path = '{}Session{}/opus/'.format(compressed_dir, sess)
        orig_wav_files = os.listdir(wav_file_path)
        for orig_wav_file in tqdm(orig_wav_files):
            orig_wav_vector, _sr = librosa.load(wav_file_path + orig_wav_file, sr=sr)
            orig_wav_file, file_format = orig_wav_file.split('.')
            for index, row in labels_df[labels_df['wav_file'].str.contains(orig_wav_file)].iterrows():
                start_time, end_time, truncated_wav_file_name, emotion, val, act, dom = row['start_time'], row['end_time'], row['wav_file'], row['emotion'], row['val'], row['act'], row['dom']
                audio_vectors[truncated_wav_file_name] = orig_wav_vector
        with open(config.AUDIO_VECT +'audio_vectors_compressed_{}.pkl'.format(session), 'wb') as f:
            pickle.dump(audio_vectors, f)


if __name__ == '__main__':
    for session in range(1,6):
        run(session)


