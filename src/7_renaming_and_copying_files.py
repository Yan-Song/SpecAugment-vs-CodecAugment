import os
import shutil
import random
import config

data_dir = config.TRAINING_DATA_DIR

codecAugmented_dir = config.CODEC_TRAINING_DIR
IEMOCAP_train_dir = config.INPUT_DIR+ 'IEMOCAP_train_dir'
img_source_dir = data_dir
img_source_dir_codec = codecAugmented_dir

if not os.path.exists(IEMOCAP_train_dir):
    os.makedirs(IEMOCAP_train_dir)

if not os.path.exists(IEMOCAP_train_dir + '/anger'):
    os.makedirs(IEMOCAP_train_dir+'/anger')
if not os.path.exists(IEMOCAP_train_dir +'/sadness'):
    os.makedirs(IEMOCAP_train_dir + '/sadness')
if not os.path.exists(IEMOCAP_train_dir +'/happiness'):
    os.makedirs(IEMOCAP_train_dir + '/happiness')
if not os.path.exists(IEMOCAP_train_dir +'/neutral'):
    os.makedirs(IEMOCAP_train_dir + '/neutral')


'''
Here in this script we made a folder inside the input folder which will be having all the taining data
The original spectogram filenames are renamed by adding _original 
The codecAugmented spectogram filenames are renamed by adding _codec
then both folders are copied to the main training dataset folder which we created above. 
'''


for subdirs in os.listdir(img_source_dir):
    for files in os.listdir(img_source_dir + '/' + subdirs):
        base = img_source_dir + '/' + subdirs + '/' + files
        old_filename =  os.path.splitext(base)[0]
        new_filename = old_filename + '_original'
        os.rename(base,new_filename + '.jpg')


for subdirs in os.listdir(img_source_dir_codec):
    for files in os.listdir(img_source_dir_codec + '/' + subdirs):
        base = img_source_dir_codec + '/' + subdirs + '/' + files
        old_filename =  os.path.splitext(base)[0]
        new_filename = old_filename + '_codec'
        os.rename(base,new_filename + '.jpg')


for subdirs in os.listdir(img_source_dir_codec):
    for files in os.listdir(img_source_dir_codec + '/' + subdirs):
        source = img_source_dir_codec + '/' + subdirs + '/' + files
        destination = IEMOCAP_train_dir + '/' + subdirs + '/'
        dest = shutil.copy(source,destination)


for subdirs in os.listdir(img_source_dir):
    for files in os.listdir(img_source_dir + '/' + subdirs):
        source = img_source_dir + '/' + subdirs + '/' + files
        destination = IEMOCAP_train_dir + '/' + subdirs + '/'
        dest = shutil.copy(source,destination)