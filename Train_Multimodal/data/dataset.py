import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import os
import os.path
from copy import copy
import numpy as np
from tqdm import tqdm
from PATH import PATH
PRESET_VARS = PATH()
import pickle

class DatasetFactory:
    def __init__(self):
        pass
    @staticmethod
    def get_by_name(dataset_name, seq_len, fps, window_size, sr, train_mode='Train', transform = None):
        if dataset_name == 'Mixed_EXPR':
            from data.dataset_Mixed_EXPR import dataset_Mixed_EXPR
            dataset = dataset_Mixed_EXPR(seq_len, fps, window_size, sr, train_mode, transform)
        elif dataset_name == 'Mixed_AU':
            from data.dataset_Mixed_AU import dataset_Mixed_AU
            dataset = dataset_Mixed_AU(seq_len, fps, window_size, sr,train_mode, transform)
        elif dataset_name == 'Mixed_VA':
            from data.dataset_Mixed_VA import dataset_Mixed_VA
            dataset = dataset_Mixed_VA(seq_len, fps, window_size, sr,train_mode, transform)
        else:
            raise ValueError("Dataset [%s] not recognized." % dataset_name)

        print('Dataset {} has been created'.format(dataset.name))
        return dataset


class DatasetBase(data.Dataset):
    def __init__(self, train_mode='Train', transform=None):
        super(DatasetBase, self).__init__()
        self._name = 'BaseDataset'
        self._root = None
        self._transform = None
        self._train_mode = None
        self._create_transform()

        self._IMG_EXTENSIONS = [
            '.jpg', '.JPG', '.jpeg', '.JPEG',
            '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
        ]

    @property
    def name(self):
        return self._name

    @property
    def path(self):
        return self._root

    def _create_transform(self):
        self._transform = transforms.Compose([])

    def get_transform(self):
        return self._transform

    def _is_image_file(self, filename):
        return any(filename.endswith(extension) for extension in self._IMG_EXTENSIONS)

    def _is_csv_file(self, filename):
        return filename.endswith('.csv')
    def __len__(self):
        return self._dataset_size

    def _read_path_label(self, file_path, task_name):
        data = pickle.load(open(file_path, 'rb'))
        data = data['{}_Set'.format(task_name)]
        # read frames ids
        if self._train_mode == 'Train':
            data = data['Train_Set']
        elif self._train_mode == 'Validation':
            data = data['Validation_Set']
        else:
            raise ValueError("train mode must be in : Train, Validation")

        return data

    def _read_dataset_paths(self, task_name):
        self._data = self._read_path_label(PRESET_VARS.Aff_wild2.data_file, task_name)
        #sample them 
        seq_len = self.seq_len
        fps = self.fps
        self.sample_seqs = []
        stride = 30 // fps
        N = seq_len
        print("Loading video frames to dataset...")
        for video in tqdm(self._data.keys(), total = len(self._data.keys())):
            data = self._data[video]
            data['video'] = [video] * len(data) # add the video name to dataframe
            N_offset = stride # if stride=1, no offset. 
            for offset in range(N_offset):
                sampled_frames = copy(data.iloc[offset:].iloc[::stride]).reset_index()
                for i in range(len(sampled_frames)//N):
                    start, end = i*N, i*N + seq_len
                    if end >= len(data):
                        start, end = len(data) - seq_len, len(data)
                    new_df = sampled_frames.iloc[start:end]
                    if not len(new_df) == seq_len:
                        assert len(new_df) < seq_len
                        count = seq_len - len(new_df)
                        for _ in range(count):
                            new_df = new_df.append(new_df.iloc[-1])
                    assert len(new_df) == seq_len
                    self.sample_seqs.append(new_df)
        self._ids = np.arange(len(self.sample_seqs)) 
        self._dataset_size = len(self._ids)
