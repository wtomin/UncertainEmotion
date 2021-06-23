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
#from utils.transforms import get_meltransform
import numpy as np
class dataset_Mixed_EXPR(DatasetBase):
    def __init__(self,  time_length,  sr = 16000, train_mode='Train', transform = None, ):
        super(dataset_Mixed_EXPR, self).__init__(time_length, sr ,train_mode, transform)
        self._name = 'dataset_Mixed_EXPR'
        self._train_mode = train_mode
        self.time_length = time_length
        self.sr = sr
        if transform is not None:
            self._transform = transform  
        else:
            self._transform = lambda x: x
        # read dataset
        self._read_dataset_paths('EXPR')
    def __getitem__(self, index):
        assert (index < self._dataset_size)
        # start_time = time.time()
        data = self.sample_collections[index]
        audio, label, audio_file, length = data
        # change the audio signals to melspectrogram
        if not torch.is_tensor(audio):
            audio = torch.tensor(audio.astype(np.float32))
        # # pack data
        sample = {'audio': audio,
                  'label': label,
                  'length': length,
                  'audio_file': audio_file,
                  'video': audio_file.split('/')[-1].split('.')[0],
                  'index': index,
                  }
        return sample



