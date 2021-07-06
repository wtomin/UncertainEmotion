import os.path
import torchvision.transforms as transforms
from .dataset import DatasetBase
from PIL import Image
import random
import numpy as np
import pickle
import pandas as pd
from utils.audio import read_audio
from PATH import PATH
import torch
PRESET_VARS = PATH()

class dataset_Mixed_VA(DatasetBase):
    def __init__(self, seq_len, fps, window_size, sr, train_mode='Train', transform = None):
        super(dataset_Mixed_VA, self).__init__(train_mode, transform)
        self._name = 'dataset_Mixed_VA'
        self._train_mode = train_mode
        self.seq_len = seq_len
        self.fps = fps
        self.window_size = int(window_size * sr)
        self.sr = sr
        if transform is not None:
            self._transform = transform  
        else:
            self._transform = lambda x: x
        # read dataset
        self._read_dataset_paths('VA')
    def _get_all_label(self):
        return self._data['label']
    def __getitem__(self, index):
        assert (index < self._dataset_size)
        # start_time = time.time()
        images = []
        labels = []
        img_paths = []
        frames_ids = []
        video_names = []
        df = self.sample_seqs[index]
        for i, row in df.iterrows():
            img_path = row['path']
            if not os.path.exists(img_path):
                par_pardir = os.path.dirname(os.path.dirname(img_path))
                img_path = img_path.replace(par_pardir, PRESET_VARS.face_dir)# replace the "*/cropped_aligned/" by the PRESET_VARS.face_dir
                assert os.path.exists(img_path)
            image = Image.open(img_path).convert('RGB')
            image = self._transform(image)
            label = row[PRESET_VARS.Aff_wild2.categories['VA']].values.astype(np.float32)
            frame_id = row['frames_ids']
            images.append(image)
            labels.append(label)
            img_paths.append(img_path)
            frames_ids.append(frame_id)
            video_names.append(row['video'])
        # pack data
        assert len(np.unique(video_names)) ==1, "the sequence must be sampled from the same video file"
        video_name = np.unique(video_names)[0]
        if '_left' in video_name:
            video_name = video_name[:-5]
        if '_right' in video_name:
            video_name = video_name[:-6]
        audio_file = os.path.join(PRESET_VARS.audio_dir, '{}.wav'.format(video_name))
        assert os.path.exists(audio_file), "audio file {} does not exist".format(audio_file)
        fps = np.unique(df['fps'].values)[0]
        offset = int(max(0, frames_ids[0]*self.sr *(1/fps)-self.window_size//2))
        num_frames = int(frames_ids[-1]*self.sr *(1/fps) + self.window_size//2)
        out, sr = read_audio(audio_file, offset=offset, num_frames=num_frames)
        out = out[0] # mono
        assert sr == self.sr, "audio sample rate must be {}".format(self.sr)
        if not self.window_size < out.size(-1):
            #skip this sample
            print("resample one sample because of the misalignment between video and audio data")
            redindex = np.random.randint(len(self))
            return self.__getitem__(redindex)
        audio_frames= []
        audio_length = []
        for i_video_frame in frames_ids:
            middle = (i_video_frame* (1/fps) +  (i_video_frame+1)*(1/fps)) *0.5
            middle = max(0, middle*sr - offset)
            start, end = max(0, middle - self.window_size//2), middle + self.window_size//2
            start, end = int(start), int(end)
            N = end - start - self.window_size
            end = end - N 
            if end > out.size(-1):
                start, end = out.size(-1) - self.window_size, out.size(-1)
            audio_frames.append(out[start:end])
            audio_length.append(out[start:end].size()[0])
        audio_frames = torch.stack(audio_frames, dim=0)
        audio_length = np.array(audio_length)
        assert audio_frames.size(0) == len(images)
        # pack data
        sample = {'image': torch.stack(images,dim=0),
                  'label': np.array(labels),
                  'audio': audio_frames,
                  'audio_length': audio_length,
                  'path': img_paths,
                  'index': index,
                  'id':frames_ids,
                  'video': np.unique(video_names)[0],
                  'fps':fps
                  }
        
        return sample
