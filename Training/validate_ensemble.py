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

class Validator(object):
    def __init__(self):
        PRESET_VARS = PATH()
        self._models = []
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
        self.validation_dataloaders = Multitask_DatasetDataLoader(
            train_mode = 'Validation', num_threads = args.n_threads_test, dataset_names=args.dataset_names,
            tasks = args.tasks, batch_size = 1, seq_len = args.seq_len, fps = 30, # validation set always sample by 30 fps
            transform = test_transforms(args.image_size))
        self.validation_dataloaders = self.validation_dataloaders.load_multitask_val_test_data()
        print("Validation sets")
        
        for task in args.tasks:
            if task in self.validation_dataloaders.keys():
                data_loader = self.validation_dataloaders[task]
                print("{}: {} images".format(task, len(data_loader)*args.batch_size * args.seq_len))
        self._validate()
    def _validate(self):
        val_start_time = time.time()
        eval_per_task = {}
        tasks = copy(args.tasks)
        if 'FA' in tasks:
            tasks.remove('FA')
        hiddens = None
        video_name = None
        optimal_Ts = {}
        for task in tasks:
            track_val_preds = {}
            track_val_labels = {}
            val_errors = OrderedDict()
            data_loader = self.validation_dataloaders[task]
            for i_val_batch, val_batch in tqdm(enumerate(data_loader), total = len(data_loader)):
                wrapped_v_batch = {task: val_batch}
                if video_name is None:
                    video_name = val_batch['video'][0]
                else:
                    if video_name != val_batch['video'][0]:
                        hiddens = None
                for i_model, model in enumerate(self._models):
                    model.set_input(wrapped_v_batch, input_tasks = [task])
                    outputs, errors, hiddens = model.forward(return_estimates=False, 
                        input_tasks = [task], FA_teacher=None, hiddens = hiddens)
                    if i_model not in track_val_preds.keys():
                        track_val_preds[i_model] = []
                        track_val_labels[i_model] = []
                    track_val_preds[i_model].append(outputs[task][task])
                    track_val_labels[i_model].append(wrapped_v_batch[task]['label'])
                # if i_val_batch>100:
                #     break
            for i_model in track_val_preds.keys():
                track_val_preds[i_model] = np.concatenate(track_val_preds[i_model], axis=0)
                track_val_labels[i_model] = np.concatenate(track_val_labels[i_model], axis=0)
            # records predictions for each model
            preds_total = []
            for i_model in track_val_preds.keys():
                if i_model not in optimal_Ts.keys():
                    optimal_Ts[i_model] = {}
                preds, labels = track_val_preds[i_model], track_val_labels[i_model]
                B, N = preds.shape[:2]
                metric_func = self._models[i_model].get_metrics_per_task()[task]
                estimates = self._models[i_model]._format_estimates({task: torch.FloatTensor(track_val_preds[i_model])})
                eval_items, eval_res = metric_func(estimates[task].reshape(B*N, -1).squeeze(), copy(labels.reshape(B*N, -1).squeeze()))
                print("** Model {} {} metric: {}".format(i_model, task, eval_res))
                optimal_T = self.optimal_T_single_model(preds, labels, task)
                optimal_Ts[i_model][task] = optimal_T
                preds_total.append(preds)

            # ensemble
            preds_total = np.array(preds_total)
            # before apply temperatures
            B, N = preds_total.shape[1:3]
            estimates = self._models[0]._format_estimates({task: torch.FloatTensor(preds_total.mean(0))})
            eval_items, eval_res = metric_func(estimates[task].reshape(B*N, -1).squeeze(), copy(labels.reshape(B*N, -1).squeeze()))
            print("# ensemble model averaged without temperature {} metric: {}".format(task, eval_res))
            Ts = np.array([optimal_Ts[i_model][task] for i_model in optimal_Ts.keys()])
            if task == 'EXPR':
                Ts = Ts[:, np.newaxis, np.newaxis, np.newaxis]
                preds_total = preds_total/Ts
            elif task == 'AU':
                Ts = Ts[:, np.newaxis, np.newaxis, : ]
                preds_total = preds_total/Ts
            elif task == 'VA':
                Ts = Ts[:, np.newaxis, np.newaxis, : ]
                preds_total[..., :20] = preds_total[..., :20]/Ts[..., [0]]
                preds_total[..., 20:] = preds_total[..., 20:]/Ts[..., [1]]

            estimates = self._models[0]._format_estimates({task: torch.FloatTensor(preds_total.mean(0))})
            eval_items, eval_res = metric_func(estimates[task].reshape(B*N, -1).squeeze(), copy(labels.reshape(B*N, -1).squeeze()))
            print("# ensemble mdoel averaged with temperature {} metric: {}".format(task, eval_res))

    def optimal_T_single_model(self, preds, labels, task):
        # test time cross validation
        N = len(labels)
        temperatures = {}
        optimal_Ts = []
        for i in range(5):
            indexes = np.random.choice(np.arange(N), size = N//2, replace=False)
            mask = np.array([False]*N)
            mask[indexes] = True 
            trainset = preds[mask], labels[mask]
            testset = preds[~mask], labels[~mask]
            optimal_T, NLL, test_NLL = self.search_T(trainset, testset, task)
            print("Fold {} for {}, optimal T: {}, minimal train NLL: {}, test NLL: {}".format(i, task, optimal_T, NLL, test_NLL))
            optimal_Ts.append(optimal_T)
        optimal_Ts = np.array(optimal_Ts)
        print("\n")
        return optimal_Ts.mean(0)

    def search_T(self, trainset, testset, task, T_range_exp = [-1, 3], N = 20):
        preds, labels = trainset
        test_preds, test_labels = testset
        if task == 'AU' or task == 'VA':
            length = labels.shape[-1]
            L = 20 if task == 'VA' else 1
        else:
            length = 1
            L = 7
        optimal_Ts = []
        optimal_nlls = []
        test_nlls = []
        for i_c in range(length):
            p = preds[..., i_c*L:(i_c+1)*L]
            l = labels[..., i_c] if length!=1 else labels
            tp = test_preds[..., i_c*L:(i_c+1)*L]
            tl = test_labels[..., i_c] if length!=1 else test_labels
            Ts_exp = np.linspace(T_range_exp[0], T_range_exp[1], num=N)
            Ts = 10**Ts_exp
            train_nlls = []
            for T in Ts:
                probas, estimates = self.get_proba_estimates(copy(p), task, T)
                train_nll = self.get_NLL(copy(probas).reshape((-1, L)).squeeze(), copy(l).reshape((-1,)), task)
                train_nlls.append(train_nll)
            train_nlls = np.array(train_nlls)
            optimal_T = Ts[train_nlls.argmin()]
            optimal_Ts.append(optimal_T)
            optimal_nlls.append(train_nlls.min())
            probas, estimates = self.get_proba_estimates(copy(tp), task, T)
            test_nll = self.get_NLL(copy(probas).reshape((-1, L)).squeeze(), copy(tl).reshape((-1,)), task)
            test_nlls.append(test_nll)
        return np.array(optimal_Ts).squeeze(), np.array(optimal_nlls).squeeze(), np.array(test_nlls).squeeze()
            
    def get_NLL(self, probas, labels, task):
        assert len(labels.shape) == 1
        if task == 'AU':
            nll = -(labels*np.log(probas + EPS) + (1-labels)*np.log(1-probas + EPS))
            return np.mean(nll)
        elif task == 'EXPR':
            nll = - np.log(probas[labels] + EPS)
            return nll.mean()
        elif task == 'VA':
            edges = np.linspace(-1, 1, 20)
            labels = np.digitize(labels, edges, right=True)
            nll = - np.log(probas[labels] + EPS)
            return nll.mean()

    def get_proba_estimates(self, preds, task, T):
        if task == 'AU':
            z = preds/T
            p = sigmoid(z)
            o = (p>0.5).astype(np.int)
        elif task == 'EXPR':
            z = preds / T
            p = softmax(z, axis=-1)
            o = p.argmax(-1)
        elif task == 'VA':
            v = preds
            v = v/T
            p = softmax(v, axis=-1)
            bins = np.linspace(-1, 1, num=20)
            o = (bins * p).sum(-1)
        return p, o


if __name__=="__main__":
    Validator()






            
        
