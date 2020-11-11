import os
import numpy as np
import shutil
import random
from shutil import copyfile
import config

original_spectogram_dir = config.ORIGINAL_SPEC_DIR
train_test_folder = original_spectogram_dir + 'train_test_folder'

codec_spectogram_dir = config.CODECAUG_SPEC_DIR
train_test_folder_codec = codec_spectogram_dir + 'train_test_folder'


if not os.path.exists(train_test_folder):
    os.makedirs(train_test_folder)
else:
    if not os.path.exists(train_test_folder + '/train'):
        os.makedirs(train_test_folder +'/train')
    if not os.path.exists(train_test_folder +'/test'):
        os.makedirs(train_test_folder +'/test')


img_source_dir = original_spectogram_dir
train_size = 0.8

subdirs = [subdir for subdir in os.listdir(img_source_dir) if os.path.isdir(os.path.join(img_source_dir, subdir))]
subdirs.remove('train_test_folder')

random.seed(42)
for subdir in subdirs:
    random.seed(42)
    subdir_fullpath = os.path.join(img_source_dir, subdir)
        
    train_subdir = os.path.join(train_test_folder + '/train', subdir)
    test_subdir = os.path.join(train_test_folder + '/test', subdir)
        
    if not os.path.exists(train_subdir):
        os.makedirs(train_subdir)

    if not os.path.exists(test_subdir):
        os.makedirs(test_subdir)

    train_counter = 0
    test_counter = 0
        
    for filename in os.listdir(subdir_fullpath):
        if filename.endswith(".jpg") or filename.endswith(".png"): 
            fileparts = filename.split('.')

            if random.uniform(0,1) <= train_size:
                copyfile(os.path.join(subdir_fullpath, filename), os.path.join(train_subdir, str(fileparts[0]) + '.' + fileparts[1]))
                train_counter += 1
            else:
                copyfile(os.path.join(subdir_fullpath, filename), os.path.join(test_subdir, str(fileparts[0]) + '.' + fileparts[1]))
                test_counter += 1
                    
    print('Copied ' + str(train_counter) + ' images to' + '{}/train/'.format(train_test_folder) + subdir)
    print('Copied ' + str(test_counter) + ' images to' + '{}/test/'.format(train_test_folder) + subdir)


'''
This code is copied 2 times 
1 for train test split of original spectograms 
2 for train test split of codecAugmented spectograms
in the next step we will add both directorie's training datasets
'''




if not os.path.exists(train_test_folder_codec):
    os.makedirs(train_test_folder_codec)
else:
    if not os.path.exists(train_test_folder_codec + '/train'):
        os.makedirs(train_test_folder_codec +'/train')
    if not os.path.exists(train_test_folder_codec +'/test'):
        os.makedirs(train_test_folder_codec +'/test')


img_source_dir_codec = codec_spectogram_dir
train_size = 0.8

subdirs = [subdir for subdir in os.listdir(img_source_dir_codec) if os.path.isdir(os.path.join(img_source_dir_codec, subdir))]
subdirs.remove('train_test_folder') #remove train test split folder from the sub directories

random.seed(42)
for subdir in subdirs:
    random.seed(42)
    subdir_fullpath = os.path.join(img_source_dir_codec, subdir)
        
    train_subdir = os.path.join(train_test_folder_codec + '/train', subdir)
    test_subdir = os.path.join(train_test_folder_codec + '/test', subdir)
        
    if not os.path.exists(train_subdir):
        os.makedirs(train_subdir)

    if not os.path.exists(test_subdir):
        os.makedirs(test_subdir)

    train_counter = 0
    test_counter = 0
        
    for filename in os.listdir(subdir_fullpath):
        if filename.endswith(".jpg") or filename.endswith(".png"): 
            fileparts = filename.split('.')

            if random.uniform(0,1) <= train_size:
                copyfile(os.path.join(subdir_fullpath, filename), os.path.join(train_subdir, str(fileparts[0]) + '.' + fileparts[1]))
                train_counter += 1
            else:
                copyfile(os.path.join(subdir_fullpath, filename), os.path.join(test_subdir, str(fileparts[0]) + '.' + fileparts[1]))
                test_counter += 1
                    
    print('Copied ' + str(train_counter) + ' images to' + '{}/train/'.format(train_test_folder_codec) + subdir)
    print('Copied ' + str(test_counter) + ' images to' + '{}/test/'.format(train_test_folder_codec) + subdir)