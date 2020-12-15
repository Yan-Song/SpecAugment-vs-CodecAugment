import librosa
import numpy as np
import librosa.display
import scipy
import scipy.io.wavfile as sf
from SpecAugment import spec_augment_pytorch
#X, sample_rate = librosa.load('/home/mds-student/Downloads/OAF_youth_happy.wav', res_type='kaiser_fast', sr = 22050*2, offset = 0.5)
#X1, sample_rate = librosa.load('Assets/Ses01F_impro05_F015.opus', res_type='kaiser_fast', sr = 22050*2, offset = 0.5)

#print(X.shape)
#print(X1.shape)
#a = np.zeros(1)
#print(a.shape)
#X2 = np.concatenate((X,a),axis = 0)
#print(X2.shape)
#error = np.subtract(X2,X1)
#print(error)
#res = np.asarray(error)
#res = res.astype('float32')
#X, sample_rate = librosa.load(path, res_type='kaiser_fast', duration = 2.5, sr = 22050*2, offset = 0.5)
X, sample_rate = librosa.load('Assets/Ses01F_impro05_F015.wav', res_type='kaiser_fast', sr = 22050*2, offset = 0.5)
import matplotlib.pyplot as plt
S1 = librosa.feature.melspectrogram(y = X, sr = 44100, n_mels=256,hop_length=128,fmax=8000)
warped_masked_spectrogram = spec_augment_pytorch.spec_augment(mel_spectrogram=S1)
S_21 = librosa.power_to_db(warped_masked_spectrogram, ref=np.max)
plt.figure(figsize=(12,4))
librosa.display.specshow(S_21, sr=44100, x_axis='time', y_axis='mel')
plt.show()

'''

S1 = librosa.feature.melspectrogram(y = res, sr = 44100, n_mels=256,hop_length=128,fmax=8000)
S_21 = librosa.power_to_db(S1, ref=np.max)
s21 = np.mean(librosa.power_to_db(S1, ref=np.max),axis = 0)
#print(s21)
plt.figure(figsize=(12,4))
librosa.display.specshow(S_21, sr=44100, x_axis='time', y_axis='mel')
plt.show()

print('S1.shape', S1.shape)
print(S1)

audio_signal = librosa.feature.inverse.mel_to_audio(S1, sr=44100, n_fft=2048, hop_length=128, window=scipy.signal.hanning)
print(audio_signal, audio_signal.shape)

sf.write('test.wav', 44100, audio_signal)
'''