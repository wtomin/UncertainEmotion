# calculate class co-ocurrence 
# A[i, j] = <Y[i], Y[j]>/|Yi|*|Yj|
import time
from options.test_options import TestOptions
from data.custom_dataset_data_loader import Multitask_DatasetDataLoader
from data.dataset import DatasetFactory
from models import ModelsFactory
from collections import OrderedDict
import os
import numpy as np
from PATH import PATH
import pandas as pf
from copy import deepcopy, copy
import pandas as pd
from tqdm import tqdm
import pickle
from utils.transforms import test_transforms, train_transforms
from utils.validation import sigmoid, softmax
import matplotlib
import matplotlib.pyplot as plt
from netcal.scaling import TemperatureScaling
from netcal.metrics import ECE 
#################RuntimeError: received 0 items of ancdata ###########################
import torch
torch.multiprocessing.set_sharing_strategy("file_system")
################# RuntimeError: unable to open shared memory object </torch_29841_2933458171> in read-write mode ############
import sys
import torch
from torch.utils.data import dataloader
from torch.multiprocessing import reductions
from multiprocessing.reduction import ForkingPickler
default_collate_func = dataloader.default_collate

def default_collate_override(batch):
    dataloader._use_shared_memory = False
    return default_collate_func(batch)

setattr(dataloader, 'default_collate', default_collate_override)

for t in torch._storage_classes:
  if sys.version_info[0] == 2:
    if t in ForkingPickler.dispatch:
        del ForkingPickler.dispatch[t]
  else:
    if t in ForkingPickler._extra_reducers:
        del ForkingPickler._extra_reducers[t]
#########################################################################
args = TestOptions().parse()

if args.auxillary:
    from utils.misc import mobile_facenet
    print("Training model with an auxillary task: face alignment.")
    args.tasks = args.tasks + ['FA']
    FA_teacher = mobile_facenet(pretrained=True, cuda=args.cuda)
    FA_teacher.eval()
else:
    FA_teacher = None

EPS = 1e-8
PRESET_VARS = PATH()
# For validation, we evaluate the emotion metrics for single models and ensemble
# we evaluate the uncertainty metrics for single models and ensemble: NLL and _extra_reducers
# visualize

class Validator(object):
    def __init__(self):
        
        self._models = []
        # load checkpooints
        num_params = 0
        for load_epoch, name in zip(args.load_epochs, args.names):
            args.load_epoch = load_epoch
            args.name = name
            model = ModelsFactory.get_by_name(args,
                        is_train= False,
                        dropout = 0.5,
                        pretrained=False)
            num_params += sum(p.numel() for p in model._model.parameters())
            self._models.append(model)
        print("{} models, total number of parameters: {}".format(len(args.load_epochs), num_params))
        # to load train(val) datasets and train(val) dataloaders. No shuffle, no data augmentation
        self.train_datasets, self.val_datasets = {}, {}
        for i, dataset_name in enumerate(args.dataset_names):
            task = args.tasks[i]
            self.train_datasets[task] = DatasetFactory.get_by_name(dataset_name, 64, 30, 'Train', test_transforms(args.image_size))
            self.val_datasets[task] = DatasetFactory.get_by_name(dataset_name, 64, 30, 'Validation', test_transforms(args.image_size))

        self.train_dataloaders, self.val_dataloaders = {}, {}
        for i, dataset_name in enumerate(args.dataset_names):
            task = args.tasks[i]
            self.train_dataloaders[task] = torch.utils.data.DataLoader(
                        self.train_datasets[task],
                        batch_size=1 ,
                        shuffle= False,
                        num_workers=int(args.n_threads_train), 
                        drop_last = False)
            self.val_dataloaders[task] = torch.utils.data.DataLoader(
                        self.val_datasets[task],
                        batch_size=1 ,
                        shuffle= False,
                        num_workers=int(args.n_threads_train), 
                        drop_last = False)

        predictions_file = 'N=5_student_round_4/new_annotation.pkl'
        self.create_dir(predictions_file)
        if not os.path.exists(predictions_file):
            new_annotation = self.eval_epoch()
            pickle.dump(new_annotation, open(predictions_file, 'wb'))
        else:
            new_annotation= pickle.load(open(predictions_file, 'rb'))
        matrix_file = "N=5_student_round_4/affinity_matrix.pkl"
        self.create_dir(matrix_file)
        if os.path.exists(matrix_file):
            affinity_matrix, tasks = pickle.load(open(matrix_file, 'rb'))
        else:
            affinity_matrix, tasks = self.cal_matrix(new_annotation) 
            pickle.dump([affinity_matrix, tasks], open(matrix_file, 'wb'))       
        self.plot_matrix(affinity_matrix, tasks)
    def create_dir(self, filepath):
        if not os.path.exists(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))
    def plot_matrix(self, affinity_matrix, tasks):
        plt.imshow(affinity_matrix)
        ticks = []
        labels = []
        id = 0
        for task in tasks:
            if task == 'EXPR':
                ticks.append(id)
                ticks.append(id+6)
                labels.append('EXPR0')
                labels.append('EXPR6')
                id+=7
            elif task == 'AU':
                ticks.append(id)
                ticks.append(id+11)
                labels.append('AU0')
                labels.append('AU11')
                id+=12
            elif task == 'VA':
                ticks.append(id)
                ticks.append(id+19)
                labels.append('V0')
                labels.append('V19')
                id+=20
                ticks.append(id)
                ticks.append(id+19)
                labels.append('A0')
                labels.append('A19')
                id+=20

        plt.title("Affinity Matrix")
        plt.xticks(ticks, labels, rotation='vertical')
        plt.yticks(ticks, labels)
        plt.colorbar()
        plt.show()
    def get_categories_per_model(self, task, i_model):
        if task == 'AU':
            categories = PRESET_VARS.Aff_wild2.categories['AU']
        elif task == 'EXPR':
            categories = PRESET_VARS.Aff_wild2.categories['EXPR']
        elif task == 'VA':
            categories = ['V{:02d}'.format(i) for i in range(20)] + ['A{:02d}'.format(i) for i in range(20)]
        categories = [cate+"_{:02d}".format(i_model) for cate in categories]
        return categories

    def cal_matrix(self, track_preds):
        total_probas = []
        track_preds_train = track_preds['Train_Set']
        track_preds_train.update(track_preds['Validation_Set'])
        track_preds = track_preds_train
        videos = track_preds.keys()
        N_models = len(self._models)
        total_probas = []
        tasks = copy(args.tasks)
        if 'FA' in tasks:
            tasks.remove('FA')
        total_probas = []
        for video in videos:
            video_df = track_preds[video]
            tasks_probas = []
            for task in tasks:
                averaged_probas = []
                for i_model in range(N_models):
                    categories = self.get_categories_per_model(task, i_model)
                    assert not video_df[categories].isnull().values.any(), "NaN in prediction!"
                    logits = video_df[categories].values
                    probas = self.logits_2_probas(logits, task)
                    averaged_probas.append(probas)
                averaged_probas = np.stack(averaged_probas, axis=0).mean(0)
                tasks_probas.append(averaged_probas)
            tasks_probas = np.concatenate(tasks_probas, axis=-1)
            total_probas.append(tasks_probas)

        total_probas = np.concatenate(total_probas, axis=0)
        dim = total_probas.shape[-1]
        affinity_matrix = np.zeros((dim, dim))
        for i in tqdm(range(dim)):
            for j in range(dim):
                if i!=j:
                    a, b = total_probas[:, i], total_probas[:, j]
                    affinity_matrix[i, j] = self.affinity_func(a, b)
        return affinity_matrix, tasks
    def affinity_func(self, a, b):
    	return np.dot(a, b)/(np.linalg.norm(a) * np.linalg.norm(b))

    def probas_2_estimates(self, probas, task):
        if task == 'EXPR':
            est = probas.argmax(axis=-1).astype(np.int)
        elif task =='AU':
            est = (probas > 0.5).astype(np.int)
        elif task == 'VA':
            v, a = probas[..., :20], probas[..., 20:]
            bins = np.linspace(-1, 1, 20)
            v = (bins * v).sum(-1)
            a = (bins * a).sum(-1)
            est = np.stack([v, a], axis = -1)
        return est

    def logits_2_probas(self, preds, task, T=1):
        preds = preds/T
        if task == 'EXPR':
            p = softmax(preds, axis=-1)
        elif task =='AU':
            p = sigmoid(preds)
        elif task =='VA':
            p = [softmax(preds[..., :20], axis=-1), softmax(preds[..., 20:], axis=-1)]
            p = np.concatenate(p, axis=-1)
        return p

    def eval_dataloader(self, dataloader, annotation_data, task):
        preds_dict = {}
        video_names = []
        frames_ids = []
        paths = []
        labels = []
        video_name = None
        hiddens = None
        for i_batch, batch_data in tqdm(enumerate(dataloader), total = len(dataloader)):
            if (video_name is None) or (not video_name is None and batch_data['video'][0] != video_name):
                video_name = batch_data['video'][0]
                hiddens = dict([(i_model, None) for i_model in range(len(self._models))])
            frames_ids.append(torch.tensor(batch_data['id']).numpy())
            paths.append(np.array(batch_data['path']).squeeze())
            assert int(paths[-1][-1].split('/')[-1].split(".")[0]) - 1 == frames_ids[-1][-1], "frames id does not match with frame path"
            labels.append(batch_data['label'][0].numpy())
            video_names.append(video_name)
            batch_data = {task: batch_data}
            for i_model, model in enumerate(self._models):
                model.set_input(batch_data, input_tasks = [task])
                outputs, _, hiddens[i_model] = model.forward(return_estimates=False, 
                    hiddens=hiddens[i_model], input_tasks = [task])
                if i_model not in preds_dict.keys():
                    preds_dict[i_model] = []
                preds_dict[i_model].append(outputs[task])
            # if i_batch>50:
            #     break 
        assert len(video_names) == len(frames_ids) and len(frames_ids) == len(paths)
        unique_video_names = np.unique(video_names)
        for video_name in tqdm(unique_video_names, total = len(unique_video_names)):
            # if video_name == '325':
            #     import pdb; pdb.set_trace()
            indexes = [index for index, video in enumerate(video_names) if video==video_name]
            video_paths = [path for index, path in enumerate(paths) if index in indexes ]
            video_frames_ids = [ids for index, ids in enumerate(frames_ids) if index in indexes ]
            video_labels = [label for index, label in enumerate(labels) if index in indexes ]
            video_preds_dict = {}
            for i_model in range(len(self._models)):
                video_preds_dict[i_model] = [preds for index, preds in enumerate(preds_dict[i_model]) if index in indexes ]
            video_df = self.format_video_df(video_paths, video_frames_ids, video_labels, video_preds_dict, task)
            if video_name not in annotation_data.keys():
                annotation_data[video_name] = video_df
            else:
                #assert len(video_df) == len(annotation_data[video_name]), "length does not match: dataframes"
                # the same video appears in another subset, so we only need to update the labels
                df1 = annotation_data[video_name]
                df2 = video_df
                merged = self.merge_two_dfs(df1, df2, on ='frames_ids')
                annotation_data[video_name] = merged
    def merge_two_dfs(self, df1, df2, on = 'frames_ids'):
        # merge two dfs on the colunmn 'frames_ids'
        df1_keys = list(df1.keys())
        df2_keys = list(df2.keys())
        df2_values = df2[on].values
        df1_values = df1[on].values
        additive_rows = [index for index, v in enumerate(df2_values) if v not in df1_values]
        #additive_on_values = [v for index, v in enumerate(df2_values) if v not in df1_values]
        part_df2 = df2.loc[additive_rows]
        merged = df1.append(part_df2, ignore_index=True)
        merged = merged.sort_values(by = [on])
        merged_values = merged[on].values
        # fill in new columns in df2
        addition_keys = [key for key in df2_keys if key not in df1_keys] 
        for key in addition_keys:
            mask = np.zeros_like(df2_values).astype(np.bool)
            mask[additive_rows] = True
            for value in df2_values[~mask]:
                replace_key_value = df2[df2[on] == value][key].values[0]
                assert (not np.isnan(replace_key_value)), "NaN in replace key value"
                if len(merged.loc[merged[on] == value, key]) > 0:
                    assert len(merged.loc[merged[on] == value, key]) == 1, "index found exceeds 1"
                    assert np.isnan(merged.loc[merged[on] == value, key].values), "should be NaN"
                    merged.loc[merged[on] == value, key] = replace_key_value
        return merged
    def format_video_df(self, video_paths, video_frames_ids, video_labels, video_preds_dict, task):
        video_paths = np.concatenate(video_paths, axis=0)
        video_frames_ids = np.concatenate(video_frames_ids, axis = 0)
        video_labels = np.concatenate(video_labels, axis=0)
        annotation_dict = {}
        for i_model in video_preds_dict.keys():
            preds_keys = list(video_preds_dict[i_model][0].keys())
            length = len(video_preds_dict[i_model])
            video_preds_dict[i_model] = dict([(key, 
                np.concatenate([video_preds_dict[i_model][i][key].squeeze() for i in range(length)], axis=0)) for key in preds_keys])
        # TODO: accelerate the speed
        unique_frames_ids = np.unique(video_frames_ids)
        mask = np.zeros_like(video_frames_ids)
        mask[np.unique(video_frames_ids, return_index=True)[1]] = True
        mask = mask.astype(np.bool) #
        unique_paths = video_paths[mask]
        unique_labels = video_labels[mask]
        unique_preds_dict = {}
        for i_model in video_preds_dict.keys():
            if i_model not in unique_preds_dict.keys():
                unique_preds_dict[i_model] = {}
            for t in video_preds_dict[i_model].keys():
                unique_preds_dict[i_model][t] = video_preds_dict[i_model][t][mask]

        annotation_dict['frames_ids'] = unique_frames_ids
        annotation_dict['path'] = unique_paths
        total_length = len(unique_paths)
        categories = PRESET_VARS.Aff_wild2.categories[task]
        if task=='EXPR':
            annotation_dict['emotion'] = unique_labels
        elif task == 'VA':
            annotation_dict['Valence'] = unique_labels[:,0]
            annotation_dict['Arousal'] = unique_labels[:,1]
        else:
            for i_c, cate in enumerate(categories):
                annotation_dict[cate] = unique_labels[:, i_c]
        for i_model in range(len(self._models)):
            preds_multitask = video_preds_dict[i_model]
            for task in ['EXPR', 'AU']:
                preds = preds_multitask[task]
                categories = PRESET_VARS.Aff_wild2.categories[task]
                assert preds.shape[-1] == len(categories)
                for i_c, cate in enumerate(categories):
                    annotation_dict[cate+"_{:02d}".format(i_model)] = preds[:, i_c]
            task = 'VA'
            preds = preds_multitask[task]
            assert preds.shape[-1] == 40
            categories = ['V{:02d}'.format(i) for i in range(20)] + ['A{:02d}'.format(i) for i in range(20)]
            for i_c, cate in enumerate(categories):
                annotation_dict[cate+"_{:02d}".format(i_model)] = preds[:, i_c]
        for key in annotation_dict.keys():
            assert len(annotation_dict[key]) == total_length, "length does not match"
        return pd.DataFrame.from_dict(annotation_dict)

    def initiate_categories(self,  task):
        new_df = {'frames_ids': [] , 'path':[]}
        # name to each category
        categories = PRESET_VARS.Aff_wild2.categories[task]
        if task !='EXPR': 
            update_dict = dict([(cate, []) for cate in categories])
        else:
            update_dict = {'emotion': []}
        new_df.update(update_dict)
        N_models = len(self._models)
        if task !='VA':
            temp_categories = categories
        else:
            temp_categories = ['V{:02d}'.format(i) for i in range(20)] + ['A{:02d}'.format(i) for i in range(20)]
        update_dict = []
        for i in range(N_models):
            update_dict += [(cate+"_{:02d}".format(i), []) for cate in temp_categories]
        update_dict = dict(update_dict)
        new_df.update(update_dict)
        return new_df
    
    def eval_epoch(self):
        new_annotation = {} # annotations is a dictionary: Train_Set and Validation_Set
        new_annotation['Train_Set'] = {}
        new_annotation['Validation_Set'] = {}
        # 'set' -> 'video_name' -> dataframe: ['frames_ids', 'path']  
        # + one of ['AU1' ...,'AU25'], ['emotion'], ['Valence', 'Arousal'], representing the true label, +
        # + ['AU1_00' ...,'AU25_00', 'Neutral_00', ..., 'Surprise_00', 'V00_00', ..., 'V19_00', 'A00_00',..., 'A19_00'], presenting the first model's logits (not the probability prediction)
        # + ['AU1_01' ...,'AU25_02', 'Neutral_02', ..., 'Surprise_02', 'V00_02', ..., 'V19_02', 'A00_02',..., 'A19_02'], presenting the second model's logits (not the probability prediction)
        # ... ...
        tasks = list(self.train_dataloaders.keys())
        for task in tasks:
            print("Evaluate on {} dataset".format(task))
            print("Train Set")
            self.eval_dataloader(self.train_dataloaders[task], new_annotation['Train_Set'], task)
            print("Validation Set")
            self.eval_dataloader(self.val_dataloaders[task], new_annotation['Validation_Set'], task)
        return new_annotation


if __name__=="__main__":
    Validator()






            
        

