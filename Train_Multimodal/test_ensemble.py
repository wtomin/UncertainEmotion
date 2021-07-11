import time
from options.test_options import TestOptions
#from data.custom_dataset_data_loader import Multitask_DatasetDataLoader
from data.test_video_dataset import Test_dataset
from models import ModelsFactory
from collections import OrderedDict
import os
import numpy as np
from PATH import PATH
import pickle
from copy import deepcopy, copy
import pandas as pd
from tqdm import tqdm
import pickle
from utils.transforms import test_transforms
from utils.validation import sigmoid, softmax

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
if 'FA' in args.auxillary:
    from utils.misc import mobile_facenet
    print("Training model with an auxillary task: face alignment.")
    args.tasks = args.tasks + ['FA']
    FA_teacher = mobile_facenet(pretrained=True, cuda=args.cuda)
    FA_teacher.eval()
else:
    FA_teacher = None
if 'VAD' in args.auxillary:
    from utils.misc import VAD_MarbleNet
    args.tasks = args.tasks + ['VAD']
    print("Training model with an auxillary task: voice activity detection.")
    VAD_teacher = VAD_MarbleNet.from_pretrained(model_name="vad_marblenet")
    if args.cuda:
        VAD_teacher.cuda()
    VAD_teacher.eval()
else:
    VAD_teacher = None
EPS = 1e-8
PRESET_VARS = PATH()
save_dir = 'single_preds_student_round_2_exp_5'
class Tester(object):
    def __init__(self):
        self.save_dir = save_dir
        self._models = []
        num_params = 0
        for load_epoch, name in zip(args.load_epochs, args.names):
            args.load_epoch = load_epoch
            args.name = name
            model = ModelsFactory.get_by_name(args,
                        is_train= False,
                        dropout = 0.5,
                        pretrained=True)
            num_params += sum(p.numel() for p in model._model.parameters())
            self._models.append(model)
        print("{} models, total number of parameters: {}".format(len(args.load_epochs), num_params))
        test_data_file = PRESET_VARS.Aff_wild2.test_data_file
        self.test_data_file = pickle.load(open(test_data_file, 'rb'))

        self._test()

    def test_one_video(self, model,  data_loader, task = 'AU'):
        track_val = {'outputs':[], 'estimates':[], 'frames_ids':[]}
        hiddens = None
        for i_val_batch, val_batch in tqdm(enumerate(data_loader), total = len(data_loader)):
            # evaluate model
            wrapped_v_batch = {task: val_batch}
            model.set_input(wrapped_v_batch, input_tasks = [task])
            outputs, _, _ = model.forward(return_estimates=False, 
                input_tasks = [task], FA_teacher=None, hiddens = hiddens)
            estimates, _, hiddens = model.forward(return_estimates=True, 
                input_tasks = [task], FA_teacher=None, hiddens = hiddens)
            #store the predictions and labels
            B, N, C = outputs[task][task].shape
            track_val['outputs'].append(outputs[task][task].reshape(B*N, C))
            track_val['frames_ids'].append(np.array([np.array(x) for x in val_batch['id']]).reshape(B*N, -1).squeeze())
            track_val['estimates'].append(estimates[task][task].reshape(B*N, -1).squeeze())
             
        for key in track_val.keys():
            track_val[key] = np.concatenate(track_val[key], axis=0)
        return track_val
    
    def save_to_file(self, frames_ids, predictions, save_path, task= 'AU'):
        save_path =os.path.join(self.save_dir, save_path)
        save_dir = os.path.dirname(os.path.abspath(save_path))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        categories = PRESET_VARS.Aff_wild2.categories[task]
        #filtered out repeated frames
        mask = np.zeros_like(frames_ids, dtype=bool)
        mask[np.unique(frames_ids, return_index=True)[1]] = True
        frames_ids = frames_ids[mask]
        predictions = predictions[mask]
        assert len(frames_ids) == len(predictions)
        with open(save_path, 'w') as f:
            f.write(",".join(categories)+"\n")
            for i, line in enumerate(predictions):
                if isinstance(line, np.ndarray):
                    digits = []
                    for x in line:
                        if isinstance(x, float):
                            digits.append("{:.4f}".format(x))
                        elif isinstance(x, np.int64):
                            digits.append(str(x))
                    line = ','.join(digits)+'\n'
                elif isinstance(line, np.int64):
                    line = str(line)+'\n'
                if i == len(predictions)-1:
                    line = line[:-1]
                f.write(line)
    def _test(self):
        
        tasks = copy(args.tasks)
        if 'FA' in tasks:
            tasks.remove('FA')
        if 'VAD' in tasks:
            tasks.remove("VAD")
        outputs_record = {}
        estimates_record = {}
        frames_ids_record = {}
        for i_model, model in enumerate(self._models):
            outputs_record[i_model] = {}
            estimates_record[i_model] = {}
            frames_ids_record[i_model] = {}
            for task in tasks:
                task_data_file = self.test_data_file[task+"_Set"]['Test_Set']
                outputs_record[i_model][task] = {}
                estimates_record[i_model][task] = {}
                frames_ids_record[i_model][task] = {}
                for i_video, video in enumerate(task_data_file.keys()):
                    video_data = task_data_file[video]
                    test_dataset = Test_dataset(64, 30, args.window_size, args.sr, video_data, task,
                        transform=test_transforms(args.image_size))
                    test_dataloader = torch.utils.data.DataLoader(
                    test_dataset,
                    batch_size= 1,
                    shuffle= False,
                    num_workers=int(args.n_threads_test),
                    drop_last=False)
                    track = self.test_one_video(model, test_dataloader, task = task)
                    
                    torch.cuda.empty_cache() 
                    outputs_record[i_model][task][video] = track['outputs']
                    estimates_record[i_model][task][video] = track['estimates']
                    frames_ids_record[i_model][task][video] = track['frames_ids']
                    print("Model ID {} Task {} Current {}/{}".format(i_model, task, i_video, len(task_data_file.keys())))
                    save_path = '{}/{}/{}.txt'.format(i_model, task, video)
                    self.save_to_file(track['frames_ids'], track['estimates'], save_path, task=task)
        
        #merge the raw outputs 
        for task in tasks:
            for video in outputs_record[0][task].keys():
                preds = []
                for i in range(len(outputs_record.keys())):
                    preds.append(outputs_record[i][task][video])
                preds = np.array(preds)
                #assert frames_ids_record[0][task][video] == frames_ids_record[1][task][video]
                video_frames_ids = frames_ids_record[0][task][video] 
                if task == 'AU':
                    merged_preds = sigmoid(preds)
                    merged_preds = np.mean(merged_preds, axis=0) 
                    merged_preds = merged_preds > (np.ones_like(merged_preds)*0.5)
                    merged_preds = merged_preds.astype(np.int64)
                    save_path = '{}/{}/{}.txt'.format('merged', task, video)
                    self.save_to_file(video_frames_ids, merged_preds, save_path, task='AU')
                elif task == 'EXPR':
                    merged_preds = softmax(preds, axis=-1).mean(0).argmax(-1).astype(np.int).squeeze()
                    save_path = '{}/{}/{}.txt'.format('merged',task,  video)
                    self.save_to_file(video_frames_ids, merged_preds, save_path, task='EXPR')
                else:
                    N = 20
                    v = softmax(preds[:, :, :N], axis=-1)
                    a = softmax(preds[:, :, N:], axis=-1)
                    bins = np.linspace(-1, 1, num=20)
                    v = (bins * v).sum(-1)
                    a = (bins * a).sum(-1)
                    merged_preds = np.stack([v.mean(0), a.mean(0)], axis = 1).squeeze() 
                    save_path = '{}/{}/{}.txt'.format( 'merged',task, video) 
                    self.save_to_file(video_frames_ids, merged_preds, save_path, task='VA')  


if __name__=="__main__":
    Tester()
