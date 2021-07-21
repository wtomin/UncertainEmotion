import time
import argparse
import os
import glob
from tqdm import tqdm
import torchaudio
import pickle
import pandas as pd
from utils import read_audio
parser = argparse.ArgumentParser()

parser.add_argument("--video_dir", type=str, 
    default= '/media/Samsung/Aff-wild2-Challenge/videos')
parser.add_argument("--audio_outdir", type=str, 
    default="audios")
parser.add_argument('--annot_dir', type=str,
    default='../Training/annotations')
parser.add_argument('--annot_file', type=str,
    default='annotations/annotations_audio.pkl')

args = parser.parse_args()
video_ext = ['.mp4', '.MP4', '.avi']
sample_rate = 16000

# given an input video, -vn remove video, -ar sample rate 16000, -ac mono, -acodec pcm_s16le for wav file
cmd = 'ffmpeg -i {} -vn -acodec pcm_s16le -ar {} -ac 1 {}'

if not os.path.exists(args.audio_outdir):
    os.makedirs(args.audio_outdir)
    for subdir in ['batch1', 'batch2']:
        video_dir = os.path.join(args.video_dir, subdir)
        for file in os.listdir(video_dir):
            if any([e in file for e in video_ext]):
                file_path = os.path.join(video_dir, file)
                output_audio = os.path.join(args.audio_outdir, file.split('.')[0]+'.wav')
                os.system(cmd.format(file_path,sample_rate, output_audio))

data_file = {}

for task in ['EXPR_Set', 'VA_Set']:
    data_file[task] = {}
    for mode in ['Train_Set', 'Validation_Set']:
        txt_files = glob.glob(os.path.join(args.annot_dir, task, mode, '*.txt'))

        audio_files = []
        lengths = []
        for txt_file in tqdm(txt_files):
            name = os.path.basename(txt_file).split('.')[0]
            if 'left' in name:
            	name = name[:-5]
            elif 'right' in name:
            	name = name[:-6]
            audio_file = os.path.join(args.audio_outdir, name+'.wav')
            assert os.path.exists(audio_file), "{} does not exist".format(audio_file)
            out, sr = read_audio(audio_file)
            length = out.shape[1]
            audio_files.append(audio_file)
            lengths.append(length)
        data_file[task][mode] = pd.DataFrame({'audio':audio_files,
            'length':lengths, 'annotation': txt_files})
pickle.dump(data_file, open(args.annot_file, 'wb'))

    



