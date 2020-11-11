import pandas as pd 
import numpy as np
import os,sys,glob,pickle
import librosa
from tqdm import tqdm 
import config
import argparse

'''
While running the script memory requirement will be increase drastically. It is advised to 
run the script for each session separately. 

For that, run the script like 
> python -W ignore src/3_pickle_to_df.py --session 1
(for all 5 session)
'''

emotion_dict = {'ang': 0,'hap': 1,'exc': 2,'sad': 3,'fru': 4,'fea': 5,'sur': 6,'neu': 7,
                'xxx': 8,'oth': 8,'dis': 8}

emotion_full_dict = {'neu': 'neutral', 'ang': 'anger', 'hap': 'happiness', 'exc': 'excited', 
                    'sad': 'sadness', 'fru':'frustrated', 'fea': 'fear', 'sur': 'surprised', 
                     'xxx': 'others', 'oth': 'others', 'dis': 'others'}


def run(session):
    labels_df = pd.read_csv(config.DF_IEMOCAP + 'df_iemocap_{}.csv'.format(session))
    iemocap_dir = config.IEMOCAP_DIR
    save_dir = config.AUDIO_VECT
    audio_vectors_path= save_dir + 'audio_vectors_opus_' 

    pickle_to_df = pd.DataFrame(columns=["filename", "audio_vector"])

    for sess in [session]:
        audio_vectors = pickle.load(open('{}{}.pkl'.format(audio_vectors_path, sess), 'rb'))
        for index, row in tqdm(labels_df[labels_df['wav_file'].str.contains('Ses0{}'.format(sess))].iterrows()):
            wav_file_name = (row['wav_file'])
            y = audio_vectors[wav_file_name]
            list_y = list(y)
            pickle_to_df = pickle_to_df.append({'filename': wav_file_name, 'audio_vector': list_y}, ignore_index=True)
    ''' 
    for our model we are taking 4 emotions i.e. Neutral, Anger, Happy and Sadness
    '''
    labels_df_subset = labels_df[labels_df["emotion"].isin(["neu", 'ang', 'hap', 'sad'])]
    labels_df_subset = labels_df_subset[["wav_file", "emotion"]]
    labels_df_subset = labels_df_subset.rename(columns={"wav_file": "filename"})
    pickle_to_subset = labels_df_subset.merge(pickle_to_df, on="filename",right_index=True)

    pickle_to_subset.to_csv(config.PICKLE_TO_SUBSET + 'df_to_subset_opus_{}.csv'.format(session), index = False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--session",
        type = int
    )
    args = parser.parse_args()
    run(session=args.session)

