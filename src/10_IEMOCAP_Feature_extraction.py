import numpy as np
import pandas as pd 
import librosa
import os, glob, sys
import config
import pickle
from tqdm import tqdm


emotion_dict = {'ang': 0,'hap': 1,'exc': 2,'sad': 3,'fru': 4,'fea': 5,'sur': 6,'neu': 7,
                'xxx': 8,'oth': 8,'dis': 8}

emotion_full_dict = {'neu': 'neutral', 'ang': 'anger', 'hap': 'happiness', 'exc': 'excited', 
                    'sad': 'sadness', 'fru':'frustrated', 'fea': 'fear', 'sur': 'surprised', 
                     'xxx': 'others', 'oth': 'others', 'dis': 'others'}


def original_melfature_extraction(session,audio_dir):
    labels_df = pd.read_csv(config.DF_IEMOCAP + 'df_iemocap_{}.csv'.format(session))
    iemocap_dir = config.IEMOCAP_DIR
    original_dir = audio_dir
    sr = 44100 #sample rate 
    audio_features = pd.DataFrame(columns = ['feature'])
    counter = 0
    emotions = []
    for sess in [session]:
        wav_file_path = '{}Session{}/wav/'.format(original_dir, sess)
        orig_wav_files = os.listdir(wav_file_path)
        for orig_wav_file in tqdm(orig_wav_files):
            orig_wav_vector, sample_rate = librosa.load(wav_file_path + orig_wav_file, sr=sr)
            sample_rate = np.array(sample_rate)
            melspect = np.mean(librosa.feature.melspectrogram(y = orig_wav_vector, sr = sample_rate, n_mels=256,hop_length=128,fmax=8000), axis = 0)
            audio_features.loc[counter] = [melspect]
            counter = counter+1
        audio_features = (pd.DataFrame(audio_features['feature']. values.tolist())).fillna(0)
        for orig_wav_file in tqdm(orig_wav_files):
            orig_wav_file, file_format = orig_wav_file.split('.')
            for index, row in labels_df[labels_df['wav_file'].str.contains(orig_wav_file)].iterrows():
                label= row['emotion']
                emotions.append(label)
        audio_features['emotions'] = pd.Series(emotions)
        if not os.path.exists('input/mel_features/Original_features'):
            os.makedirs('input/mel_features/Original_features')
        audio_feature_subset = audio_features[audio_features["emotions"].isin(["neu", 'ang', 'hap', 'sad'])]
        audio_feature_subset.to_csv('input/mel_features/Original_features/audio_features_{}.csv'.format(session), index = False)


def codec_melfeature_extraction(session,audio_dir):
    labels_df = pd.read_csv(config.DF_IEMOCAP + 'df_iemocap_{}.csv'.format(session))
    iemocap_dir = config.IEMOCAP_DIR
    compressed_dir = audio_dir
    sr = 44100 #sample rate 
    audio_features = pd.DataFrame(columns = ['feature'])
    counter = 0
    emotions = []
    for sess in [session]:
        wav_file_path = '{}Session{}/opus/'.format(compressed_dir, sess)
        orig_wav_files = os.listdir(wav_file_path)
        for orig_wav_file in tqdm(orig_wav_files):
            orig_wav_vector, sample_rate = librosa.load(wav_file_path + orig_wav_file, sr=sr)
            sample_rate = np.array(sample_rate)
            melspect = np.mean(librosa.feature.melspectrogram(y = orig_wav_vector, sr = sample_rate, n_mels=256,hop_length=128,fmax=8000), axis = 0)
            audio_features.loc[counter] = [melspect]
            counter = counter+1
        audio_features = (pd.DataFrame(audio_features['feature']. values.tolist())).fillna(0)
        for orig_wav_file in tqdm(orig_wav_files):
            orig_wav_file, file_format = orig_wav_file.split('.')
            for index, row in labels_df[labels_df['wav_file'].str.contains(orig_wav_file)].iterrows():
                label= row['emotion']
                emotions.append(label)
        audio_features['emotions'] = pd.Series(emotions)
        if not os.path.exists('input/mel_features/CodecAugmented_features'):
            os.makedirs('input/mel_features/CodecAugmented_features')
        audio_feature_subset = audio_features[audio_features["emotions"].isin(["neu", 'ang', 'hap', 'sad'])]
        audio_feature_subset.to_csv('input/mel_features/CodecAugmented_features/audio_features_{}.csv'.format(session), index = False)

def specAugmented_melfeature_extraction(session,audio_dir):
    from SpecAugment import spec_augment_pytorch
    labels_df = pd.read_csv(config.DF_IEMOCAP + 'df_iemocap_{}.csv'.format(session))
    iemocap_dir = config.IEMOCAP_DIR
    compressed_dir = audio_dir
    sr = 44100 #sample rate 
    audio_features = pd.DataFrame(columns = ['feature'])
    counter = 0
    emotions = []
    for sess in [session]:
        wav_file_path = '{}Session{}/wav/'.format(compressed_dir, sess)
        orig_wav_files = os.listdir(wav_file_path)
        for orig_wav_file in tqdm(orig_wav_files):
            orig_wav_vector, sample_rate = librosa.load(wav_file_path + orig_wav_file, sr=sr)
            sample_rate = np.array(sample_rate)
            melspect = librosa.feature.melspectrogram(y = orig_wav_vector, sr = sample_rate, n_mels=256,hop_length=128,fmax=8000)
            warped_masked_spectrogram = np.mean((spec_augment_pytorch.spec_augment(mel_spectrogram=melspect)),axis = 0)
            audio_features.loc[counter] = [warped_masked_spectrogram]
            counter = counter+1
        audio_features = (pd.DataFrame(audio_features['feature']. values.tolist())).fillna(0)
        for orig_wav_file in tqdm(orig_wav_files):
            orig_wav_file, file_format = orig_wav_file.split('.')
            for index, row in labels_df[labels_df['wav_file'].str.contains(orig_wav_file)].iterrows():
                label= row['emotion']
                emotions.append(label)
        audio_features['emotions'] = pd.Series(emotions)
        audio_feature_subset = audio_features[audio_features["emotions"].isin(["neu", 'ang', 'hap', 'sad'])]
        if not os.path.exists('input/mel_features/SpecAugmented_features'):
            os.makedirs('input/mel_features/SpecAugmented_features')
        audio_feature_subset.to_csv('input/mel_features/SpecAugmented_features/audio_features_{}.csv'.format(session), index = False)



    
if __name__ == '__main__':
    for session in range(1,6):
        codec_melfeature_extraction(session,audio_dir = 'input/Compressed_audio/IEMOCAP_Compressed_br64_fs5/')
        #specAugmented_melfeature_extraction(session, audio_dir= config.ORIGINAL_WAVFILE)
        original_melfature_extraction(session, audio_dir= config.ORIGINAL_WAVFILE)
