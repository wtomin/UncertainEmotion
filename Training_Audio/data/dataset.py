import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import os
import os.path
from copy import copy
import numpy as np
from tqdm import tqdm
from PATH import PATH
from utils import read_audio, downsample2, upsample2, read_Expr, read_VA, resample_labels
PRESET_VARS = PATH()
import pickle

class DatasetFactory:
    def __init__(self):
        pass
    @staticmethod
    def get_by_name(dataset_name, seq_len, sr, train_mode='Train', transform = None):
        if dataset_name == 'Mixed_EXPR':
            from data.dataset_Mixed_EXPR import dataset_Mixed_EXPR
            dataset = dataset_Mixed_EXPR(seq_len, sr, train_mode, transform)
        elif dataset_name == 'Mixed_VA':
            from data.dataset_Mixed_VA import dataset_Mixed_VA
            dataset = dataset_Mixed_VA(seq_len, sr, train_mode, transform)
        else:
            raise ValueError("Dataset [%s] not recognized." % dataset_name)

        print('Dataset {} has been created'.format(dataset.name))
        return dataset

def frames_to_label(label_array, audio_frames, discard_value,
    video_fps =30, audio_sr = 16000):
    snippet_length = audio_sr / video_fps # a label corresponds to a snippet of audio frames
    assert np.abs(snippet_length - audio_frames.size(-1)/len(label_array)) < 5
    to_drops = label_array == discard_value
    to_save = []
    for to_drop in to_drops:
        to_save.extend([~to_drop] * int(snippet_length))
    min_length = min(audio_frames.size(-1), len(to_save))
    to_save = np.array(to_save)
    if len(to_save.shape) == 2:
        to_save_audio = np.all(to_save, axis=1)
        to_save_labels = np.all(~to_drops, axis=1)
    elif len(to_save.shape) == 1:
        to_save_audio = to_save
        to_save_labels = ~to_drops
    label_array = label_array[to_save_labels]
    audio_frames = audio_frames[:, :min_length]
    audio_frames = audio_frames[:, to_save_audio[:min_length]]
    return label_array, audio_frames

class DatasetBase(data.Dataset):
    def __init__(self, train_mode='Train', transform=None):
        super(DatasetBase, self).__init__()
        self._name = 'BaseDataset'
        self._root = None
        self._transform = None
        self._train_mode = None
        self.ref_video_fps = 30
        self._create_transform()

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
        data = self._read_path_label(PRESET_VARS.Aff_wild2.data_file, task_name)
        #sample them 
        seq_len = self.seq_len
        self.sample_seqs = []

        print("Loading audio frames to dataset...")
        for i_line, line in tqdm(data.iterrows(), total = len(data)):
            audio_file = line['audio']
            length = line['length']
            annot_file = line['annotation']
            out, sr = read_audio(audio_file)
            if sr < self.sr:
                assert (self.sr / sr)%2 == 0, "source sample rate {} must be 2^n times {}".format(sr, self.sr)
                while sr < self.sr:
                    out = upsample2(out)
                    sr = sr*2
            elif sr > self.sr:
                assert (sr / self.sr)%2==0, "source sample rate {} must be 2^n times {}".format(sr, self.sr)
                while sr > self.sr:
                    out = downsample2(out)
                    sr = int(0.5*sr)
            assert sr == self.sr, 'sample rate must be {}Hz'.format(self.sr)
            if task_name == 'VA':
                read_func = read_VA
                discard_value = -5.
            elif task_name =='EXPR':
                read_func = read_Expr
                discard_value = -1.
            labels = read_func(annot_file)
            audio_length = out.size(-1)
            time_length = audio_length/sr

            target_len = int(time_length*self.ref_video_fps)
            if target_len > 2* len(labels):
                continue
            labels = resample_labels(labels, 
                    target_len = target_len,
                    discrete=task_name == 'EXPR')

            labels, out = frames_to_label(labels, out, 
                discard_value=discard_value, video_fps = self.ref_video_fps, audio_sr = sr)
            N = sr * self.seq_len
            n = self.seq_len * self.ref_video_fps
            for i in range(out.size(-1)//N):
                start, end = i*N, (i+1)*N
                if end >= out.size(-1):
                    start, end = out.size(-1) - N, out.size(-1)
                new_audio = out[:, start:end]
                start, end = i*n, (i+1)*n 
                if end >= len(labels):
                    start, end = len(labels) - n, len(labels)
                new_labels = labels[start: end]
                assert new_audio.size(-1) == N
                assert len(new_labels) == n
                self.sample_seqs.append((new_audio, new_labels, audio_file))

        self._ids = np.arange(len(self.sample_seqs)) 
        self._dataset_size = len(self._ids)
