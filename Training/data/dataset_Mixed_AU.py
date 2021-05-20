import os.path
import torchvision.transforms as transforms
from .dataset import DatasetBase
from PIL import Image
import random
import numpy as np
import pickle
import pandas as pd
from PATH import PATH
import torch
PRESET_VARS = PATH()

class dataset_Mixed_AU(DatasetBase):
    def __init__(self, seq_len, fps, train_mode='Train', transform = None):
        super(dataset_Mixed_AU, self).__init__(train_mode, transform)
        self._name = 'dataset_Mixed_AU'
        self._train_mode = train_mode
        self.seq_len = seq_len
        self.fps = fps
        if transform is not None:
            self._transform = transform  
        else:
            self._transform = lambda x: x
        # read dataset
        self._read_dataset_paths('AU')
    def _get_all_label(self):
        return self._data['label']
    def __getitem__(self, index):
        assert (index < self._dataset_size)
        # start_time = time.time()
        images = []
        labels = []
        img_paths = []
        frames_ids = []
        df = self.sample_seqs[index]
        for i,row in df.iterrows():
            img_path = row['path']
            image = Image.open(img_path).convert('RGB')
            image = self._transform(image)
            label = row[PRESET_VARS.Aff_wild2.categories['AU']].values.astype(np.float32)
            frame_id = row['frames_ids']
            images.append(image)
            labels.append(label)
            img_paths.append(img_path)
            frames_ids.append(frame_id)
        # pack data
        sample = {'image': torch.stack(images,dim=0),
                  'label': np.array(labels),
                  'path': img_paths,
                  'index': index,
                  'id':frames_ids
                  }
        return sample


