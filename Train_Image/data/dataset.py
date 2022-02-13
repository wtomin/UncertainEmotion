import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import os
import os.path
from copy import copy
import numpy as np
import torch
import pytorch_lightning as pl
import numpy as np
from tqdm import tqdm
from PATH import PATH
PRESET_VARS = PATH()
import pickle
import pandas as pd

class DatasetFactory:
    def __init__(self):
        pass
    @staticmethod
    def get_by_name(dataset_name, seq_len, fps, train_mode='Train', transform = None):
        if dataset_name == 'Mixed_EXPR':
            from data.dataset_Mixed_EXPR import dataset_Mixed_EXPR
            dataset = dataset_Mixed_EXPR(seq_len, fps, train_mode, transform)
        elif dataset_name == 'Mixed_AU':
            from data.dataset_Mixed_AU import dataset_Mixed_AU
            dataset = dataset_Mixed_AU(seq_len, fps, train_mode, transform)
        elif dataset_name == 'Mixed_VA':
            from data.dataset_Mixed_VA import dataset_Mixed_VA
            dataset = dataset_Mixed_VA(seq_len, fps, train_mode, transform)
        else:
            raise ValueError("Dataset [%s] not recognized." % dataset_name)

        print('Dataset {} has been created'.format(dataset.name))
        return dataset


class DatasetBase(data.Dataset):
    def __init__(self, train_mode='Train', transform=None, downsample = 1):
        super(DatasetBase, self).__init__()
        self._name = 'BaseDataset'
        self._root = None
        self._transform = None
        self._train_mode = None
        self._create_transform()
        self.downsample = downsample

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
        
        self.samples = []

        print("Loading video frames to dataset...")
        for video in tqdm(self._data.keys(), total = len(self._data.keys())):
            data = self._data[video]
            data['video'] = [video] * len(data) # add the video name to dataframe
            sampled_frames = copy(data.iloc[::self.downsample]).reset_index()
            self.samples.append(sampled_frames)
        self.samples = pd.concat(self.samples, axis=0, ignore_index=True)
        
    @property
    def _ids(self):
        return np.arange(len(self.samples)) 
    @property
    def _dataset_size(self):
        return len(self._ids)

    def resample_images(self, num_samples):
        if num_samples > self._dataset_size:
            N = num_samples - self._dataset_size
            select_df = self.samples.sample(n=N)
            self.samples = pd.concat([self.samples, select_df], axis=0, ignore_index=True)
        elif num_samples < self._dataset_size:
            N = self._dataset_size -  num_samples
            indices = np.random.choice(self.samples.index, N, replace=False)
            self.samples = self.samples.drop(indices)


class dataset_Task(DatasetBase):
    def __init__(self, task, train_mode='Train', transform = None,  downsample = 1):
        super(dataset_Task, self).__init__(train_mode, transform, downsample)
        self._name = 'dataset_Mixed_{}'.format(task)
        self.task = task
        self._train_mode = train_mode

        if transform is not None:
            self._transform = transform  
        else:
            self._transform = lambda x: x
        # read dataset
        self.task = task
        self.downsample = downsample
        self._read_dataset_paths(task)
    def __getitem__(self, index):
        assert (index < self._dataset_size)
        df = self.samples.iloc[index]
        img_path = df['path']
        if not os.path.exists(img_path):
            par_pardir = os.path.dirname(os.path.dirname(img_path))
            img_path = img_path.replace(par_pardir, PRESET_VARS.face_dir)# replace the "*/cropped_aligned/" by the PRESET_VARS.face_dir
            assert os.path.exists(img_path)
        image = Image.open(img_path).convert("RGB")
        image = self._transform(image)
        if self.task =='AU':
            label = df[PRESET_VARS.Aff_wild2.categories['AU']].values.astype(np.float32)
        elif self.task == 'EXPR':
            label = df['label']
        else:
            label = df[PRESET_VARS.Aff_wild2.categories['VA']].values.astype(np.float32)
        video = df['video']
        frame_id = df['frames_ids']
        return image, label, video, frame_id

class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, datasets_dict):
        super().__init__()
        self.datasets_dict = datasets_dict
        Length = max(len(self.datasets_dict[key]) for key in self.datasets_dict.keys())
        for key in self.datasets_dict:
            self.datasets_dict[key].resample_images(Length)
        print("All datasets resampled to {} images".format(Length))

    def __getitem__(self, idx):
        return dict([(key, self.datasets_dict[key][idx]) for key in self.datasets_dict.keys()])
    def __len__(self):
        return min(len(self.datasets_dict[key]) for key in self.datasets_dict.keys())

class DataModule(pl.LightningDataModule):
    def __init__(self, tasks, transform_train, transform_test, 
        num_workers_train, num_workers_test, batch_size, downsamples = []):
        super().__init__()
        self.tasks = tasks
        self.downsamples = downsamples
        self.transform_train = transform_train
        self.transform_test = transform_test
        assert len(self.tasks) == len(self.downsamples), "#. tasks must be the same as the #. downsamples"
        self.batch_size = batch_size
        self.num_workers_train = num_workers_train
        self.num_workers_test = num_workers_test
    def setup(self, stage = None):
        if stage == 'fit' or stage is None:
            train_datasets = []
            for i, task in enumerate(self.tasks):
                emotion_dataset = dataset_Task(task, 'Train', self.transform_train,
                    self.downsamples[i])
                train_datasets.append([task, emotion_dataset])
            self.train_datasets = dict(train_datasets)
            log = "Train datasets are loaded. "
            for task in self.train_datasets.keys():
                log+= '{}: {} images '.format(task, len(self.train_datasets[task]))
            print(log)
            self.train_datasets = ConcatDataset(self.train_datasets)
            val_datasets = []
            for i, task in enumerate(self.tasks):
                emotion_dataset = dataset_Task(task, 'Validation', 
                    self.transform_test, 1) # do not downsample the validation set
                val_datasets.append(emotion_dataset)
            self.val_datasets = val_datasets
            log  = "Validation datasets are loaded. "
            for i, task in enumerate(self.tasks):
                log+= '{}: {} images '.format(task, len(self.val_datasets[i]))
            print(log)
    def train_dataloader(self):
        # before call
        return torch.utils.data.DataLoader(self.train_datasets,
            batch_size = self.batch_size,
            num_workers = self.num_workers_train,
            shuffle = True,
            drop_last = True)

    def val_dataloader(self):
        # return a 
        return [torch.utils.data.DataLoader(dataset,
            batch_size = self.batch_size, num_workers = self.num_workers_test,
            shuffle=False, drop_last = False) for dataset in self.val_datasets]

