import os.path
import torchvision.transforms as transforms
from .dataset import DatasetBase
from PIL import Image
import random
import numpy as np
from copy import copy
from utils.audio import read_audio
import pickle
import pandas as pd
from PATH import PATH
import torch
PRESET_VARS = PATH()

class Test_dataset(object):
    def __init__(self, seq_len, fps, window_size, sr, video_data, task, transform = None):
        self._name = 'Test_dataset'
        self._seq_len = seq_len
        self.window_size = int(window_size * sr)
        self.sr = sr
        if transform is not None:
            self._transform = transform
        else:
            self._transform = lambda x:x
        # read dataset
        self._data = video_data
        self.task = task
        self._read_dataset()

    def __getitem__(self, index):
        assert (index < self._dataset_size)
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
            frame_id = row['frames_ids']
            images.append(image)
            label = row[[str(i) for i in range(self._label_size)]].values.astype(np.float32)
            if self._label_size == 1:
                label = label.squeeze().astype(np.int64) #EXPR
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
        fps = np.unique(df['fps'].values)[0] +1 # to avoid too large offset
        offset = int(max(0, frames_ids[0]*self.sr *(1/fps)-self.window_size//2))
        num_frames = int(frames_ids[-1]*self.sr *(1/fps) + self.window_size//2)  - offset
        if num_frames < self.window_size:
            num_frames = self.window_size
        out, sr = read_audio(audio_file, offset=offset, num_frames=num_frames)
        out = out[0] # mono
        assert sr == self.sr, "audio sample rate must be {}".format(self.sr)
        if self.window_size > out.size(-1):
            import pdb; pdb.set_trace()
            print("resample one sample because of the misalignment between video and audio data")
            # resample this window by change the offset 
            N = self.window_size - out.size(-1)
            out, sr = read_audio(audio_file, offset=offset, num_frames=self.window_size)
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
    def _read_dataset(self):
        #sample them 
        seq_len = self._seq_len
        self.sample_seqs = []
        N = seq_len
        data = {'path': self._data['path'], 'frames_ids':self._data['frames_ids'], 'fps': self._data['fps'], 'video': self._data['video']}
        self._label_size = len(PRESET_VARS.Aff_wild2.categories[self.task]) if self.task !='EXPR' else 1
        num_images = self._data['label'].shape[0]
        data.update(dict([(str(i), np.zeros((num_images,))) \
            for i in range(self._label_size)]))
        for i in range(len(self._data['path'])//N + 1):
            start, end = i*N, i*N + seq_len
            if end >= len(self._data):
                start, end = len(self._data) - seq_len, len(self._data)
            new_df = self._data.iloc[start:end]
            if not len(new_df) == seq_len:
                assert len(new_df) < seq_len
                count = seq_len - len(new_df)
                for _ in range(count):
                    new_df = new_df.append(new_df.iloc[-1])
            assert len(new_df) == seq_len
            self.sample_seqs.append(new_df)
        self._ids = np.arange(len(self.sample_seqs)) 
        self._dataset_size = len(self._ids)
    def __len__(self):
        return self._dataset_size
