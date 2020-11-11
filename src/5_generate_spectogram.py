import pandas as pd 
import numpy as np
import os,sys,glob,pickle
import librosa
import librosa.display
from tqdm import tqdm 
import config
import argparse
import time
import multiprocessing as mp
import ast
import matplotlib.pyplot as plt


emotion_full_dict = {'neu': 'neutral', 'ang': 'anger', 'hap': 'happiness', 'exc': 'excited', 
                    'sad': 'sadness', 'fru':'frustrated', 'fea': 'fear', 'sur': 'surprised', 
                     'xxx': 'others', 'oth': 'others', 'dis': 'others'}


'''
Make sure while randomize dataframe take random_state = 50 on the line number 
This way we will match files with other foms of augmentation like SpecAugment and CodecAugment
Run this script as below
>> python -W ignore src/5_generate_spectogram.py --session 3 
'''


def run(session):

    labels_df = pd.read_csv(config.DF_IEMOCAP + 'df_iemocap_{}.csv'.format(session))
    iemocap_dir = config.IEMOCAP_DIR
    save_dir = config.AUDIO_VECT
    audio_vectors_path= save_dir + 'audio_vectors_opus_' 
    pathAudio = iemocap_dir
    pathImage = config.IMAGE_PATH

    df = pd.read_csv(config.DF_TO_SUBSET + 'df_to_subset_opus_{}.csv'.format(session))
    df = df.sample(frac=1,random_state = 50).reset_index(drop = True)
    df = df.sort_values(by = ['emotion'])
    N = df['emotion'].eq('hap').sum()
    df = df.sort_values('emotion').groupby('emotion').head(N)

    sample_rate = 44100

    for row in tqdm(df.itertuples(index=True, name='Pandas')):
        a = getattr(row, "audio_vector")
        image_fname = getattr(row, "filename")
        label = getattr(row, "emotion")
        res = ast.literal_eval(a)
        res = np.asarray(res)
        res = res.astype('float32')
        S = librosa.feature.melspectrogram(y = res, sr = sample_rate, n_mels=256,hop_length=128,fmax=8000)
        S_2 = librosa.power_to_db(S, ref=np.max)
        plt.figure(figsize=(12,4))
        librosa.display.specshow(S_2, sr=sample_rate, x_axis='time', y_axis='mel')
        fig1 = plt.gcf()
        plt.axis('off')
        save_path = pathImage + emotion_full_dict[label] + '/' + image_fname + '.jpg'
        fig1.savefig(save_path, dpi=50)
        plt.close(fig1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--session",
        type = int
    )
    args = parser.parse_args()
    run(session=args.session)

 
    

