import time
from options.test_options import TestOptions
from data.custom_dataset_data_loader import Multitask_DatasetDataLoader
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
from utils.transforms import test_transforms
from utils.validation import sigmoid, softmax
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})
from sklearn.metrics import brier_score_loss
from netcal.scaling import TemperatureScaling
from utils.Miscalibration import _Miscalibration as miscalibration
import torchvision
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

if args.auxiliary:
    from utils.misc import mobile_facenet
    print("Training model with an auxiliary task: face alignment.")
    args.tasks = args.tasks + ['FA']
    FA_teacher = mobile_facenet(pretrained=True, cuda=args.cuda)
    FA_teacher.eval()
else:
    FA_teacher = None

EPS = 1e-8
# For validation, we evaluate the emotion metrics for single models and ensemble
# we evaluate the uncertainty metrics for single models and ensemble: NLL and _extra_reducers
# visualize

class Validator(object):
    def __init__(self):
        PRESET_VARS = PATH()
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
        testset = torchvision.datasets.FashionMNIST(root='./OOD_data', train=True,
                                               download=True, transform=test_transforms(args.image_size))
        print("# Samples: {}".format(len(testset)))
        self.testloader = torch.utils.data.DataLoader(testset, batch_size= 64,
                                                 shuffle=False, num_workers=args.n_threads_test)
        save_file = 'OOD_CIFAR10/val_res.pkl'
        if not os.path.exists(os.path.dirname(save_file)):
            os.makedirs(os.path.dirname(save_file))
        if not os.path.exists(save_file):
            track_val_preds, track_val_labels = self._validate_OOD()
            pickle.dump([track_val_preds, track_val_labels], open(save_file, 'wb'))
        else:
            track_val_preds, track_val_labels = pickle.load(open(save_file, 'rb'))

    def _validate_OOD(self):
        eval_per_task = {}
        tasks = copy(args.tasks)
        if 'FA' in tasks:
            tasks.remove('FA')
        hiddens = dict([(i_model, None) for i_model in range(len(self._models))])

        track_val_preds = {}
        track_val_labels = {}
        for task in tasks:
            track_val_preds[task] = {}
            track_val_labels[task] = {}
        default_task = 'AU'
        for i_val_batch, val_batch in tqdm(enumerate(self.testloader), total = len(self.testloader)):
            
            images, labels = val_batch
            if images.size(1) == 1:
                images = images.repeat(1, 3, 1, 1)
            BS = images.size(0)
            val_batch = {'image': images.unsqueeze(1), 'label': torch.zeros((BS, 1, 12))}
            wrapped_v_batch = {default_task: val_batch}
            for i_model, model in enumerate(self._models):
                model.set_input(wrapped_v_batch, input_tasks = [default_task])
                
                outputs, errors, hiddens[i_model] = model.forward(return_estimates=False, 
                    input_tasks = [default_task], FA_teacher=None, hiddens = hiddens[i_model])
                for task in tasks:
                    if i_model not in track_val_preds[task].keys():
                        track_val_preds[task][i_model] = []
                        track_val_labels[task][i_model] = []
                    track_val_preds[task][i_model].append(outputs[default_task][task])
                    track_val_labels[task][i_model].append(wrapped_v_batch[default_task]['label'])
        for task in tasks:
            for i_model in track_val_preds[task].keys():
                track_val_preds[task][i_model] = np.concatenate(track_val_preds[task][i_model], axis=0)
                track_val_labels[task][i_model] = np.concatenate(track_val_labels[task][i_model], axis=0)
        return track_val_preds, track_val_labels

if __name__=="__main__":
    Validator()






            
        
