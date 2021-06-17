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

class Test_dataset(object):
    def __init__(self, seq_len, video_data, task, transform = None):
        self._name = 'Test_dataset'
        self._seq_len = seq_len
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
        # start_time = time.time()
        images = []
        labels = []
        img_paths = []
        frames_ids = []
        df = self.sample_seqs[index]
        for i, row in df.iterrows():
            img_path = row['path']
            image = Image.open(img_path).convert('RGB')
            image = self._transform(image)
            label = row[[str(i) for i in range(self._label_size)]].values.astype(np.float32)
            if self._label_size == 1:
                label = label.squeeze().astype(np.int64) #EXPR
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
                  'frames_ids':frames_ids
                  }
        # print (time.time() - start_time)
        return sample
    def _read_dataset(self):
        #sample them 
        seq_len = self._seq_len
        self.sample_seqs = []
        N = seq_len
        data = {'path': self._data['path'], 'frames_ids':self._data['frames_ids']} # dataframe are easier for indexing
        self._label_size = len(PRESET_VARS.Aff_wild2.categories[self.task]) if self.task !='EXPR' else 1
        num_images = self._data['label'].shape[0]
        data.update(dict([(str(i), np.zeros((num_images,))) \
            for i in range(self._label_size)]))
        self._data = pd.DataFrame.from_dict(data)
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
