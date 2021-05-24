import os.path
from .dataset import DatasetBase
from PIL import Image
import random
import numpy as np
import pickle
import pandas as pd
import torch
from PATH import PATH
PRESET_VARS = PATH()

class dataset_Mixed_EXPR(DatasetBase):
    def __init__(self, seq_len, fps, train_mode='Train', transform = None):
        super(dataset_Mixed_EXPR, self).__init__(train_mode, transform)
        self._name = 'dataset_Mixed_EXPR'
        self._train_mode = train_mode
        self.seq_len = seq_len
        self.fps = fps
        if transform is not None:
            self._transform = transform  
        else:
            self._transform = lambda x: x
        # read dataset
        self._read_dataset_paths('EXPR')
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
            image = Image.open(img_path).convert('RGB')
            image = self._transform(image)
            label = row['label']
            frame_id = row['frames_ids']
            images.append(image)
            labels.append(label)
            img_paths.append(img_path)
            frames_ids.append(frame_id)
            video_names.append(row['video'])
        
        # pack data
        assert len(np.unique(video_names)) ==1, "the sequence must be sampled from the same video file"
        sample = {'image': torch.stack(images,dim=0),
                  'label': np.array(labels),
                  'path': img_paths,
                  'index': index,
                  'id':frames_ids,
                  'video': np.unique(video_names)[0],
                  }
        # print (time.time() - start_time)
        return sample



