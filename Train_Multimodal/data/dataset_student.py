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
    def __init__(self, seq_len, fps, annotation_file, window_size, sr,train_mode='Train', transform = None):
        super(DatasetStudent, self).__init__(train_mode, transform)
        self._name = 'DatasetStudent'
        self._train_mode = train_mode
        self.seq_len = seq_len
        self.fps = fps
        self.window_size = int(window_size * sr)
        self.sr = sr
        if transform is not None:
            self._transform = transform  
        else:
            self._transform = lambda x: x
        # read dataset
        self._read_dataset_paths(annotation_file)
    def get_task_label(self, df, task):
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
                preds_i_model.append(value)
            preds_i_model = np.stack(preds_i_model, axis=-1)
            if task == 'EXPR':
                preds_i_model = softmax(preds_i_model)
            elif task == 'AU':
                preds_i_model = sigmoid(preds_i_model)
            elif task == 'VA':
                preds_i_model = np.concatenate([softmax(preds_i_model[:20]), softmax(preds_i_model[20:])], axis=-1)
            total_probas.append(preds_i_model)
        total_probas = np.stack(total_probas, axis=0).mean(0)
        return total_probas
    def __getitem__(self, index):
        assert (index < self._dataset_size)
        # start_time = time.time()
        images = []
        EXPR_labels = []
        AU_labels = []
        VA_labels = []
        img_paths = []
        frames_ids = []
        video_names = []
        df = self.sample_seqs[index]
        for i, row in df.iterrows():
            img_path = row['path']
            if not os.path.exists(img_path):
                par_pardir = os.path.dirname(os.path.dirname(img_path))
                img_path = img_path.replace(par_pardir, PRESET_VARS.face_dir)# replace the "*/cropped_aligned/" by the PRESET_VARS.face_dir
                assert os.path.exists(img_path)

            image = Image.open(img_path).convert('RGB')
            image = self._transform(image)
            label = self.get_task_label(row, 'EXPR')
            EXPR_labels.append(label)
            label = self.get_task_label(row, 'AU')
            AU_labels.append(label)
            label = self.get_task_label(row, 'VA')
            VA_labels.append(label)
            frame_id = row['frames_ids']
            images.append(image)
            img_paths.append(img_path)
            frames_ids.append(frame_id)
            video_names.append(row['video'])
        # pack data
        assert len(np.unique(video_names)) ==1, "the sequence must be sampled from the same video file"
        video_name = np.unique(video_names)[0]
        if '_left' in video_name:
            video_name = video_name[:-5]
        if '_right' in video_name:
            video_name = video_name[:-6]
        audio_file = os.path.join(PRESET_VARS.audio_dir, '{}.wav'.format(video_name))
        assert os.path.exists(audio_file), "audio file {} does not exist".format(audio_file)
        fps = np.unique(df['fps'].values)[0] +1 # to avoid too large offset
        offset = int(max(0, frames_ids[0]*self.sr *(1/fps)-self.window_size//2))
        num_frames = int(frames_ids[-1]*self.sr *(1/fps) + self.window_size//2)  - offset
        if num_frames < self.window_size:
            num_frames = self.window_size
        out, sr = read_audio(audio_file, offset=offset, num_frames=num_frames)
        out = out[0] # mono
        assert sr == self.sr, "audio sample rate must be {}".format(self.sr)
        if self.window_size > out.size(-1):
            import pdb; pdb.set_trace()
            print("resample one sample because of the misalignment between video and audio data")
            # resample this window by change the offset 
            N = self.window_size - out.size(-1)
            out, sr = read_audio(audio_file, offset=offset, num_frames=self.window_size)
        audio_frames= []
        audio_length = []
        for i_video_frame in frames_ids:
            middle = (i_video_frame* (1/fps) +  (i_video_frame+1)*(1/fps)) *0.5
            middle = max(0, middle*sr - offset)
            start, end = max(0, middle - self.window_size//2), middle + self.window_size//2
            start, end = int(start), int(end)
            N = end - start - self.window_size
            end = end - N 
            if end > out.size(-1):
                start, end = out.size(-1) - self.window_size, out.size(-1)
            audio_frames.append(out[start:end])
            audio_length.append(out[start:end].size()[0])
        audio_frames = torch.stack(audio_frames, dim=0)
        audio_length = np.array(audio_length)
        assert audio_frames.size(0) == len(images)
        # pack data
        sample = {'image': torch.stack(images,dim=0),
                  'EXPR_label': np.array(EXPR_labels).astype(np.float32),
                  'AU_label': np.array(AU_labels).astype(np.float32),
                  'VA_label': np.array(VA_labels).astype(np.float32),
                  'audio': audio_frames,
                  'audio_length': audio_length,
                  'path': img_paths,
                  'index': index,
                  'id':frames_ids,
                  'video': np.unique(video_names)[0],
                  'fps':fps
                  }
        return sample