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
import logging

class DatasetFactory:
    def __init__(self):
        pass
    @staticmethod
    def get_by_name(dataset_name, time_length, sr, train_mode='Train', transform = None):
        if dataset_name == 'Mixed_EXPR':
            from data.dataset_Mixed_EXPR import dataset_Mixed_EXPR
            dataset = dataset_Mixed_EXPR(time_length, sr, train_mode, transform)
        elif dataset_name == 'Mixed_VA':
            from data.dataset_Mixed_VA import dataset_Mixed_VA
            dataset = dataset_Mixed_VA(time_length, sr, train_mode, transform)
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
    def __init__(self, time_length, sr = 16000, train_mode='Train', transform=None):
        super(DatasetBase, self).__init__()
        self._name = 'BaseDataset'
        self._root = None
        self._transform = None
        self._train_mode = None
        self.time_length = time_length
        self.ref_video_fps = 30
        self.shift_length = 1/self.ref_video_fps
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
        
        self.sample_collections = []
        print("Loading audio frames to dataset...")
        # parse all audio files in the data 
        for i_line, line in tqdm(data.iterrows(), total = len(data)):
            audio_file = line['audio']
            length = line['length']
            annot_file = line['annotation']
            # read annotations
            if task_name == 'VA':
                read_func = read_VA
                discard_value = -5.
            elif task_name =='EXPR':
                read_func = read_Expr
                discard_value = -1.
            labels = read_func(annot_file)
            # read the audio file
            out, sr = read_audio(audio_file)
            assert sr == self.sr, "expect the sample rate equals {}, got {}".format(self.sr, sr)
            audio_length = out.size(-1)
            duration = audio_length/sr
            # upsample or downsample the labels to meet the length of audio signal
            target_len = int(duration*self.ref_video_fps)
            if target_len > 2* len(labels):
                logging.info("Skip the audio file {} because the label shortage".format(audio_file))
                continue
            labels = resample_labels(labels, 
                    target_len = target_len,
                    discrete=task_name == 'EXPR')
            # the discarded values in the labels should be filtered, so are the audio signals 
            labels, out = frames_to_label(labels, out, 
                discard_value=discard_value, video_fps = self.ref_video_fps, audio_sr = sr)
            assert out.size(0) == 1, 'expect mono audio signal'
            out = out.view(-1)
            audio_length = int(np.round(sr * self.time_length)) # 0.63 * 16000= 100080
            audio_stride = int(np.round(sr * self.shift_length))  # 1/30 * 16000 = 5333
            #label_length = int(np.round(self.time_length * self.ref_video_fps)) # 0.63 * 30 = 19
            label_stride = int(np.round(self.shift_length * self.ref_video_fps)) # 1
            assert label_stride == 1

            n_segments = labels.shape[0]
            if out.size(0) < audio_length:                
                logging.info("Skip the audio file {} because the audio is too short".format(audio_file))
                continue
            for i_seg in range(n_segments):
                l = labels[i_seg]
                start, end = i_seg * audio_stride, i_seg * audio_stride + audio_length
                if start > out.size(0):
                    raise ValueError("start index exceeds the audio signal length")
                elif end > out.size(0):
                    start, end = out.size(0) - audio_length, out.size(0)
                new_audio =  out[start: end]
                assert len(new_audio) == audio_length, "audio length incorrect"
                self.sample_collections.append((new_audio, l, audio_file, audio_length))
            # if i_line>20:
            #     break
        self._ids = np.arange(len(self.sample_collections)) 
        self._dataset_size = len(self._ids)
