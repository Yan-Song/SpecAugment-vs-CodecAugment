
import subprocess 
import os
import argparse
import config
import sys
from tqdm import tqdm

'''
Usage: opusenc [options] input_file output_file.opus

       opusenc --bitrate XX --save-range range.txt --cvbr input_file.wav output_file.opus 

Encodes input_file using Opus.
It can read the WAV, AIFF, FLAC, Ogg/FLAC, or raw files.

General options:
 -h, --help         This help
 -V, --version      Version information
 --quiet            Quiet mode

input_file can be:
  filename.wav      file
  -                 stdin

output_file can be:
  filename.opus     compressed file
  -                 stdout

Encoding options:
 --bitrate n.nnn    Target bitrate in kbit/sec (6-256/channel)
 --vbr              Use variable bitrate encoding (default)
 --cvbr             Use constrained variable bitrate encoding
 --hard-cbr         Use hard constant bitrate encoding
 
 --comp n           Encoding complexity (0-10, default: 10 (slowest))
                    Encoding computational complexity (0-10, default: 10). Zero gives the fastest encodes 
                    but lower quality, while 10 gives the highest quality but slower encoding.


 --framesize n      Maximum frame size in milliseconds
                      (2.5, 5, 10, 20, 40, 60, default: 20)
                    Smaller framesizes achieve lower latency but less quality at a given bitrate.
                    Sizes greater than 20ms are only interesting at fairly low bitrates.

 --expect-loss      Percentage packet loss to expect (default: 0)
 --downmix-mono     Downmix to mono
 --downmix-stereo   Downmix to stereo (if >2 channels)
 --max-delay n      Maximum container delay in milliseconds
                      (0-1000, default: 1000)

Diagnostic options:
 --serial n         Forces a specific stream serial number
 --save-range file  Saves check values for every frame to a file
 --set-ctl-int x=y  Pass the encoder control x with value y (advanced)
                      Preface with s: to direct the ctl to multistream s
                      This may be used multiple times


 --padding n        Extra bytes to reserve for metadata (default: 512)
 --discard-comments Don't keep metadata when transcoding
 --discard-pictures Don't keep pictures when transcoding

Input options:
 --raw              Raw input
 --raw-bits n       Set bits/sample for raw input (default: 16)
 --raw-rate n       Set sampling rate for raw input (default: 48000)
 --raw-chan n       Set number of channels for raw input (default: 2)
 --raw-endianness n 1 for bigendian, 0 for little (defaults to 0)
 --ignorelength     Always ignore the datalength in Wave headers
 

 for i in *.wav;
  do name=$(echo "$i" | cut -d'.' -f1)
  echo "$name"
  opusenc --bitrate 64 "$i" "${name}.opus"
done

here change the bitrate as below table

Media type:	                         Typical use case:	                               Recommended bit rate range:
Narrowband speech (NB)	             	   Speech-only on low-bandwidth networks	            8 to 12 kbps

Wideband speech (WB)	                  Speech on a typical network	                        16 to 20 kbps

Fullband speech (FB)	                  Speech on a good network	                           28 to 40 kbps
 
Fullband monaural music (FB mono)	   Music streaming with a stereo microphone	           48 to 64 kbps

Fullband stereo music (FB stereo)	   Music streaming	                                  64 to 128 kbps


to run this script 
>> python src/9_opus_conversion_files.py --session 2 --bitrate 28 --framesize 10 --complexity 5


'''



def run(session,bitrate,framesize,complexity):

    original_wavfiles = config.ORIGINAL_WAVFILE
    compressed_wavfiles = config.CODEC_FILE
    directory = 'IEMOCAP_Compressed_br{}_fs{}'.format(bitrate,framesize)
    compressed_files_path = os.path.join(compressed_wavfiles, directory)
    
    for sess in [session]:
        compressed_file_sessions = 'Session{}'.format(session)
        path = os.path.join(compressed_files_path,compressed_file_sessions)
        compressed_session_folder = os.makedirs(path)
        final_folder_name = 'opus'
        final_folder_path = os.path.join(path,final_folder_name)
        final_session_folder = os.makedirs(final_folder_path)
        wav_file_path = '{}Session{}/wav/'.format(original_wavfiles, sess)
        orig_wav_files = os.listdir(wav_file_path)
        for orig_wav_file in orig_wav_files:
            file_name = orig_wav_file.split('.')[0]
            file_name = os.path.basename(file_name)
            cmd = ['opusenc', '--bitrate',bitrate ,'--cvbr','--framesize',framesize, '--comp', complexity,
                    wav_file_path+orig_wav_file, 
                    '{}.opus'.format(os.path.join(compressed_files_path+ '/' +compressed_file_sessions +'/'+ final_folder_name ,file_name))]
            subprocess.call(cmd)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--framesize',type= str)
    parser.add_argument("--bitrate", type = str)
    parser.add_argument("--complexity",type=str)
    args = parser.parse_args()
    for session in range(1,6):
      run(session,bitrate=args.bitrate,framesize=args.framesize,complexity=args.complexity)
    #run(session=args.session,bitrate=args.bitrate,framesize=args.framesize,complexity=args.complexity)	

        
    


