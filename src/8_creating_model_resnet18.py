from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import IPython.display as ipd
warnings.filterwarnings('ignore')
import config
#torch
import torch
import torch.nn.functional as F
from torch import optim,nn
from torchvision import datasets,models,transforms

import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset,DataLoader
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report

data_dir = config.IMAGE_PATH
model_save_path = config.MODEL
#train_dir = '/home/mds-student/Documents/aDITYA/multimodal-speech-emotion-recognition-master/IEMOCAP_train_dir/'
#train_dir = data_dir + '/train'
test_dir = data_dir + '/test'
train_dir = '/home/mds-student/Documents/aDITYA/IEMOCAP/input/IEMOCAP_train_dir/'