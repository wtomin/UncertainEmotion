import os.path
from .dataset import DatasetBase
import torchvision.transforms as transforms
import torch.utils.data as data
from PIL import Image
import random
import numpy as np
import pickle
import pandas as pd
from PATH import PATH
import torch
PRESET_VARS = PATH()
from utils.transforms import get_meltransform
import numpy as np
class dataset_Mixed_VA(DatasetBase):
    def __init__(self,  time_length, sr = 16000, train_mode='Train', transform = None, num_downsamples = 4):
        super(dataset_Mixed_VA, self).__init__(time_length, sr, train_mode, transform)
        self._name = 'dataset_Mixed_VA'
        self._train_mode = train_mode
        self.time_length = time_length
        self.sr = sr
        #self.meltransform = get_meltransform(self.time_length, self.sr)
        if transform is not None:
            self._transform = transform  
        else:
            self._transform = lambda x: x
        self.num_downsamples = num_downsamples
        # read dataset
        self._read_dataset_paths('VA')

    def __getitem__(self, index):
        assert (index < self._dataset_size)
        # start_time = time.time()
        data = self.sample_collections[index]
        audio, label, audio_file, length = data
        # change the audio signals to melspectrogram
        if not torch.is_tensor(audio):
            audio = torch.tensor(audio.astype(np.float32))
        sample = {'audio': audio,
                  'label': label,
                  'length': length,
                  'audio_file': audio_file,
                  'video': audio_file.split('/')[-1].split('.')[0],
                  'index': index,
                  }
        return sample
