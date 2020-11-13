import pandas as pd
import numpy as np
import config
import os
import sys


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import keras
from keras import regularizers
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, Embedding, LSTM
from keras.layers import Input, Flatten, Dropout, Activation, BatchNormalization
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from keras.utils import np_utils, to_categorical
from keras.callbacks import ModelCheckpoint

def feature_dataframe_gen(feature_dir):
    feature_dataframe = []
    for sess in range(1,6) :
        feature_df = pd.read_csv('{}audio_features_{}.csv'.format(feature_dir,sess))
        feature_df = feature_df.sample(frac=1,random_state = 50).reset_index(drop = True)
        feature_df = feature_df.sort_values(by = ['emotions'])
        N = feature_df['emotions'].eq('hap').sum()
        feature_df = feature_df.sort_values('emotions').groupby('emotions').head(N)
        feature_dataframe.append(feature_df)
    feature_dataframe = (pd.concat(feature_dataframe)).fillna(0)
    return feature_dataframe


ori_feature_dataframe = feature_dataframe_gen(feature_dir = 'input/mel_features/Original_features/')

ori_feature_dataframe = ori_feature_dataframe.loc[:, (ori_feature_dataframe==0.0).mean() < .9]
#print(ori_feature_dataframe.shape)
#specAugmented_feature_dataframe = feature_dataframe_gen(feature_dir='input/mel_features/SpecAugmented_features/')


X_train_ori, X_test_ori, y_train_ori, y_test_ori = train_test_split(ori_feature_dataframe.drop('emotions',1), ori_feature_dataframe.emotions, test_size = 0.20, shuffle = True, random_state = 42)
#X_train_spec, X_test_spec, y_train_spec, y_test_spec = train_test_split(specAugmented_feature_dataframe.drop('emotions',1), specAugmented_feature_dataframe.emotions, test_size = 0.20, shuffle = True, random_state = 42)

#X_train = pd.concat([X_train_ori,X_train_spec])
#y_train = pd.concat([y_train_ori,y_train_spec])
X_train = X_train_ori
y_train = y_train_ori
X_test = X_train_ori
y_test = y_train_ori

mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)
X_train = ((X_train - mean)/std)
X_test = ((X_test - mean)/std)



X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)


lb = LabelEncoder()
y_train = np_utils.to_categorical(lb.fit_transform(y_train))
y_test = np_utils.to_categorical(lb.fit_transform(y_test))

X_train = np.expand_dims(X_train, axis = 2)
X_test = np.expand_dims(X_test, axis = 2)





model = Sequential()
model.add(Conv1D(256, 8, padding='same',input_shape=(X_train.shape[1],1)))  # X_train.shape[1] = No. of Columns
model.add(Activation('relu'))
model.add(Conv1D(256, 8, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.05))
model.add(MaxPooling1D(pool_size=(8)))
model.add(Conv1D(128, 8, padding='same'))
model.add(Activation('relu'))
model.add(Conv1D(128, 8, padding='same'))
model.add(Activation('relu'))
model.add(Conv1D(128, 8, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.10))
model.add(MaxPooling1D(pool_size=(8)))
model.add(Conv1D(64, 8, padding='same'))
model.add(Activation('relu'))
model.add(Conv1D(64, 8, padding='same'))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(4)) # Target class number
model.add(Activation('softmax'))
opt = keras.optimizers.SGD(lr=0.0001, momentum=0.0, decay=0.0, nesterov=False)
#opt = keras.optimizers.Adam(lr=0.000001)
#opt = keras.optimizers.RMSprop(lr=0.00001, decay=1e-6)
print(model.summary())

model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['accuracy'])
model_history=model.fit(X_train, y_train, batch_size=16, epochs=60, validation_data=(X_test, y_test))

model_save_dir = config.MODEL
model.save(model_save_dir)

model_json = model.to_json()
with open(model_save_dir+'/' +"model_json.json", "w") as file:
    file.write(model_json)

