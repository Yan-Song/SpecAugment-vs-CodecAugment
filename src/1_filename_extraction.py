import re
import pandas as pd 
import os
import config
''' 
For each dialogue the information is provided below. 
[START_TIME - END_TIME] TURN_NAME EMOTION [V, A, D]

[6.2901 - 8.2357]	Ses01F_impro01_F000	neu	[2.5000, 2.5000, 2.5000]
C-E2:	Neutral;	()
C-E3:	Neutral;	()
C-E4:	Neutral;	()
C-F1:	Neutral;	(curious)
A-E3:	val 3; act 2; dom  2;	()
A-E4:	val 2; act 3; dom  3;	(mildly aggravated but staying polite, attitude)
A-F1:	val 3; act 2; dom  1;	()

We are extracting the useful information like Filename, respective emotions, 

Each file provides the detailed evaluation reports for the categorical evaluators (e.g., C-E1), 
The dimensional evaluators (e.g., A-E1), and the self-evaluatiors (e.g., C-F1 or C-M1, A-F1 or A-M1). 
The utterance-level information can be found in the first line of an utterance summary.  
The first entry represents the start and end times for the utterance.  
The second entry is the utterance name (e.g., Ses01_impro01_F003).  
The third entry is the ground truth (if no majority ground truth could be assigned, the ground truth is labeled xxx).  
The final engry is the average dimensional evaluation (over the evaluators, except the self-evaluators).

These files are in IEMOCAP full release folder as in this project we just want to check results after
CodecAugment we will not addup them in input folder

'''

def run(session): 

    info_line = re.compile(r'\[.+\]\n', re.IGNORECASE)

    start_times, end_times, wav_file_names, emotions, vals, acts, doms = [], [], [], [], [], [], []

    for sess in [session]:
        emo_evaluation_dir = '/home/mds-student/Documents/aDITYA/5_spec_augment-master/IEMOCAP_full_release_withoutVideos/IEMOCAP_full_release/Session{}/dialog/EmoEvaluation/'.format(sess)
        evaluation_files = [l for l in os.listdir(emo_evaluation_dir) if 'Ses' in l]
        for file in evaluation_files:
            with open(emo_evaluation_dir + file) as f:
                content = f.read()
            info_lines = re.findall(info_line, content)
            for line in info_lines[1:]:  # the first line is a header
                start_end_time, wav_file_name, emotion, val_act_dom = line.strip().split('\t')
                start_time, end_time = start_end_time[1:-1].split('-')
                val, act, dom = val_act_dom[1:-1].split(',')
                val, act, dom = float(val), float(act), float(dom)
                start_time, end_time = float(start_time), float(end_time)
                start_times.append(start_time)
                end_times.append(end_time)
                wav_file_names.append(wav_file_name)
                emotions.append(emotion)
                vals.append(val)
                acts.append(act)
                doms.append(dom)



    df_iemocap = pd.DataFrame(columns=['start_time', 'end_time', 'wav_file', 'emotion', 'val', 'act', 'dom'])

    df_iemocap['start_time'] = start_times
    df_iemocap['end_time'] = end_times
    df_iemocap['wav_file'] = wav_file_names
    df_iemocap['emotion'] = emotions
    df_iemocap['val'] = vals
    df_iemocap['act'] = acts
    df_iemocap['dom'] = doms

    df_iemocap.to_csv(config.DF_IEMOCAP +'df_iemocap_{}.csv'.format(session),index = False)


if __name__ == '__main__':
    for session in range(1,6):
        run(session)
