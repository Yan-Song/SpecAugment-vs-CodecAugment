
# SpecAugment-vs-CodecAugment for Speech Emotion Recognition

## Introduction
Deep Learning has been applied successfully to Automatic Speech Recognition (ASR) and Speech Emotion Recognition (SER), where the main focus of research has been designing better network architectures, for example, DNNs, CNNs, RNNs and end-to-end models. However, these models tend to overfit easily and require large amounts of training data.

Let's check some open source, Speech Emotion datasets. 
* TESS (Toronto emotional speech set) 
> A set of 200 target words were spoken in the carrier phrase "Say the word _____' by two actresses (aged 26 and 64 years) and recordings were made of the set portraying each of seven emotions (anger, disgust, fear, happiness, pleasant surprise, sadness, and neutral). There are 2800 stimuli in total.

* RAVDESS (The Ryerson Audio-Visual Database of Emotional Speech and Song)
> The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS) contains 7356 files (total size: 24.8 GB). The database contains 24 professional actors (12 female, 12 male), vocalizing two lexically-matched statements in a neutral North American accent. Speech includes calm, happy, sad, angry, fearful, surprise, and disgust expressions, and song contains calm, happy, sad, angry, and fearful emotions.

* CREMA -d (crema_d)
> This data set consists of facial and vocal emotional expressions in sentences spoken in a range of basic emotional states (happy, sad, anger, fear, disgust, and neutral). 7,442 clips of 91 actors with diverse ethnic backgrounds were collected.

* SAVEE (Surrey Audio-Visual Expressed Emotion)
> This consists of recordings from 4 male actors in 7 different emotions, 480 British English utterances in total

* IEMOCAP (Interactive Emotional Dyadic Motion Capture)
> IEMOCAP database is annotated by multiple annotators into categorical labels, such as anger, happiness, sadness, neutrality, as well as dimensional labels such as valence, activation and dominance. The dataset contains useful 5531 utterances.

The main problem we saw here, is the lack of Datasets and that's why, implementation of different data augmentation techniques are necessary. 

### SpecAugment
In the absence of an adequate volume of training data, it is possible to increase the effective size of existing data through the process of [data augmentation](https://www.microsoft.com/en-us/research/wp-content/uploads/2003/08/icdar03.pdf), which has contributed to significantly improving the performance of deep networks in the domain of [image classification](https://ai.googleblog.com/2018/06/improving-deep-learning-performance.html). In the case of speech emotion recognition, augmentation traditionally involves deforming the audio waveform used for training in some fashion (e.g., by speeding it up or slowing it down), or adding background noise. This has the effect of making the dataset effectively larger, as multiple augmented versions of a single input is fed into the network over the course of training, and also helps the network become robust by forcing it to learn relevant features. However, existing conventional methods of augmenting audio input introduces additional computational cost and sometimes requires additional data.

A recent paper from Google Brain, “[SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition](https://arxiv.org/abs/1904.08779)”, thay took a new approach to augmenting audio data, treating it as a visual problem rather than an audio one. Instead of augmenting the input audio waveform as is traditionally done, SpecAugment applies an augmentation policy directly to the audio spectrogram (i.e., an image representation of the waveform). This method was simple, computationally cheap to apply, and does not require additional data. It is also surprisingly effective in improving the performance of ASR

SpecAugment modifies the spectrogram by [warping](https://www.tensorflow.org/api_docs/python/tf/contrib/image/sparse_image_warp) it in the time direction, masking blocks of consecutive frequency channels, and masking blocks of utterances in time.
As shown in below picture, 
First Picture is a typical representation of Audio file and the second picture is a log mel spectrogram of given audio.

![First Picture is a typical representation of Audio file and the second picture is a log mel spectrogram of given audio](https://3.bp.blogspot.com/-ac9sLynWUUQ/XLo-Z4TYKII/AAAAAAAAEEU/UaOV-sDGlPw6dIYo_aHwJf0rKYg1IUehgCEwYBhgL/s640/image3.png)
![The log mel spectrogram is augmented by warping in the time direction, and masking (multiple) blocks of consecutive time steps (vertical masks) and mel frequency channels (horizontal masks).](https://4.bp.blogspot.com/-joiPxVcyU-c/XLo-bKDUSvI/AAAAAAAAEEg/NhqAZtH7hxILt5et82zIrSKvPq5DHFLCgCEwYBhgL/s640/image6.png) 

### CodecAugment

In contrast to SpecAugment, we came up with another data augmentation technique, Codec Augmentation. 
#### But why Audio Encoding ?
Audio encoding is used to expand the training data set. Audio encoding is typically used to reduce the data bandwidth for storage and especially for transmission. Audio is compressed to enable efficient storage and to simplify transmission. Compression can be lossless or lossy in audio encoding. It automatically removes certain information not needed to fulfill the purpose of the application used. For music transmission, the compression is optimized to remove certain parts of the original sound signal that go beyond the auditory resolution ability. Voice quality and intelligibility are of interest for voice data transmission. Losing information while compressing audio without drastically changing it is an essential aspect of the audio augmentation method at the data level. Such loss of data later affects feature extraction and enables data variation in the training set. Several studies investigated the impact of compression on spectral quality and acoustic features but without analyzing the possibility of using these altered features for data augmentation. Although there are many ways to change the acoustic information, we have focused on the bit rate and code type as parameters.

For the codec type, we have chosen Opus, as it is an audio codec that can be of high quality and low delay with the capability of incorporating multiple channels. It supports several audio-frequency bandwidths and is suitable for various real-time audio signals, including speech and music also it is an open source and loyalty free software. Furthermore, Opus achieves an outstanding quality of experience across multiple listening tests, especially in the higher bands.
For bitrates, OPUS codec gives us 3 variations.
1. Low Pass (Nerrow band) from 8 to 16 Kbps
2. Hybrid mode (Wide band) from 16 to 64 Kbps
3. MDCT (Wide band/ psychoacoustics) above 64Kbps

We took one or two bitrates from each variations i.e. 8 Kbps, 16Kbps, 32 Kbps, 48 Kbps, 64 Kbps and 128 Kbps. 
Let's check out how Codec Augment works.   
![On the above image we have original wav file spectrogram and below image shows compressed audio file spectrogram at 8kbps with OPUS codec ](https://github.com/Aditya3107/SpecAugment-vs-CodecAugment/blob/master/Assets/collage.png)



