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
from utils.validation import sigmoid, softmax
import torch
from utils.misc import cal_uncertainty
import pandas as pd
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

    def _read_path_label(self, file_path):
        data = pickle.load(open(file_path, 'rb'))
        # read frames ids
        if self._train_mode == 'Train':
            data = data['Train_Set']
        elif self._train_mode == 'Validation':
            data = data['Validation_Set']
        else:
            raise ValueError("train mode must be in : Train, Validation")
        return data
    def _read_dataset_paths(self, annotation_file):
        self._data = self._read_path_label(annotation_file)
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

class DatasetStudent(DatasetBase):
    def __init__(self, seq_len, fps, annotation_file, train_mode='Train', transform = None):
        super(DatasetStudent, self).__init__(train_mode, transform)
        self._name = 'DatasetStudent'
        self._train_mode = train_mode
        self.seq_len = seq_len
        self.fps = fps
        if transform is not None:
            self._transform = transform  
        else:
            self._transform = lambda x: x
        # read dataset
        self._read_dataset_paths(annotation_file)
    @classmethod
    def get_task_label_and_uncertainty(self, df, task):
        if task == 'EXPR':
            categories = PRESET_VARS.Aff_wild2.categories['EXPR']
        elif task == 'AU':
            categories = PRESET_VARS.Aff_wild2.categories['AU']
        elif task == 'VA':
            categories = ['V{:02d}'.format(i) for i in range(20)] + ['A{:02d}'.format(i) for i in range(20)]
        N_models = 0
        for key in df.keys():
            if categories[0]+"_" in key:
                N_models+=1
        total_probas = []
        for i_model in range(N_models):
            preds_i_model = []
            for cate in categories:
                value = df[cate+"_{:02d}".format(i_model)]
                if isinstance(value, pd.Series):
                    value = value.values
                preds_i_model.append(value)
            preds_i_model = np.stack(preds_i_model, axis=-1)
            if task == 'EXPR':
                preds_i_model = softmax(preds_i_model)
            elif task == 'AU':
                preds_i_model = sigmoid(preds_i_model)
            elif task == 'VA':
                preds_i_model = np.concatenate([softmax(preds_i_model[..., :20]), softmax(preds_i_model[..., 20:])], axis=-1)
            total_probas.append(preds_i_model)
        total_probas = np.stack(total_probas, axis=0)
        mean_probas = total_probas.mean(0)
        if task == 'EXPR':
            alea_uncertainty, epi_uncertainty = cal_uncertainty(total_probas)
        elif task == 'VA':
            alea_uncertainty, epi_uncertainty = cal_uncertainty(total_probas[..., :20])
            alea_uncertainty = [alea_uncertainty]
            epi_uncertainty = [epi_uncertainty]
            alea_uncertainty.append(cal_uncertainty(total_probas[..., 20:])[0])
            epi_uncertainty.append(cal_uncertainty(total_probas[..., 20:])[1])
            alea_uncertainty = np.stack(alea_uncertainty, axis=-1)
            epi_uncertainty = np.stack(epi_uncertainty, axis=-1)
        elif task == 'AU':
            au_probas = np.stack([1- total_probas, total_probas], axis=-1)
            num_aus = au_probas.shape[-2]
            aleas, epis = [], []
            for i in range(num_aus):
                alea_uncertainty, epi_uncertainty = cal_uncertainty(au_probas[..., i, :])
                aleas.append(alea_uncertainty)
                epis.append(epi_uncertainty)
            alea_uncertainty = np.stack(aleas, axis=-1)
            epi_uncertainty = np.stack(epis, axis=-1)
        return mean_probas, np.stack([alea_uncertainty, epi_uncertainty], axis=-1)
    
    def __getitem__(self, index):
        assert (index < self._dataset_size)
        # start_time = time.time()
        images = []
        EXPR_labels = []
        EXPR_uncertainty= []
        AU_labels = []
        AU_uncertainty = []
        VA_labels = []
        VA_uncertainty = []
        img_paths = []
        frames_ids = []
        video_names = []
        df = self.sample_seqs[index]
        for i, row in df.iterrows():
            img_path = row['path']
            image = Image.open(img_path).convert('RGB')
            image = self._transform(image)
            label, uncertainty = self.get_task_label_and_uncertainty(row, 'EXPR')
            EXPR_labels.append(label)
            EXPR_uncertainty.append(uncertainty)
            label, uncertainty = self.get_task_label_and_uncertainty(row, 'AU')
            AU_labels.append(label)
            AU_uncertainty.append(uncertainty)
            label, uncertainty = self.get_task_label_and_uncertainty(row, 'VA')
            VA_labels.append(label)
            VA_uncertainty.append(uncertainty)
            frame_id = row['frames_ids']
            images.append(image)
            img_paths.append(img_path)
            frames_ids.append(frame_id)
            video_names.append(row['video'])
        # pack data
        assert len(np.unique(video_names)) ==1, "the sequence must be sampled from the same video file"
        sample = {'image': torch.stack(images,dim=0),
                  'EXPR_label': np.array(EXPR_labels).astype(np.float32),
                  'EXPR_uncertainty': np.array(EXPR_uncertainty).astype(np.float32),
                  'AU_label': np.array(AU_labels).astype(np.float32),
                  'AU_uncertainty': np.array(AU_uncertainty).astype(np.float32),
                  'VA_label': np.array(VA_labels).astype(np.float32),
                  'VA_uncertainty': np.array(VA_uncertainty).astype(np.float32),
                  'path': img_paths,
                  'index': index,
                  'id':frames_ids,
                  'video': np.unique(video_names)[0],
                  }

        return sample