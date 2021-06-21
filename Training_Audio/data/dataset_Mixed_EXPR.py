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
from utils.transforms import get_meltransform
import numpy as np
class dataset_Mixed_EXPR(DatasetBase):
    def __init__(self, seq_len, sr = 16000, train_mode='Train', transform = None, num_downsamples = 4):
        super(dataset_Mixed_EXPR, self).__init__(train_mode, transform)
        self._name = 'dataset_Mixed_EXPR'
        self._train_mode = train_mode
        self.seq_len = seq_len
        self.sr = sr
        self.meltransform = get_meltransform(self.seq_len, self.sr)
        if transform is not None:
            self._transform = transform  
        else:
            self._transform = lambda x: x
        self.num_downsamples = num_downsamples
        # read dataset
        self._read_dataset_paths('EXPR')
    def __getitem__(self, index):
        assert (index < self._dataset_size)
        # start_time = time.time()
        data = self.sample_seqs[index]
        audio, label, audio_file = data
        # change the audio signals to melspectrogram
        if not torch.is_tensor(audio):
            audio = torch.tensor(audio.astype(np.float32))
        audio = self.meltransform(audio).detach()
        length_labels = label.shape[0]
        target_audio_length = 2 ** self.num_downsamples * length_labels
        N_C, W, H = audio.size()
        if H > target_audio_length:
            new_audio = audio[:, :, :target_audio_length]
        elif H < target_audio_length:
            new_audio = torch.zeros((N_C, W, target_audio_length))
            new_audio[:, :, :target_audio_length] = audio
        else:
            new_audio = audio
        audio = new_audio
        # pack data
        sample = {'audio': audio,
                  'label': np.array(label),
                  'audio_file': audio_file,
                  'video': audio_file.split('/')[-1].split('.')[0],
                  'index': index,
                  }
        return sample



