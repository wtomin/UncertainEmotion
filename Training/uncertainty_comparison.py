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
res_files = ['N=5_loss_reweight/val_res.pkl', 'N=5_student_round_1/val_res.pkl',
'N=5_student_round_2/val_res.pkl', 'N=5_student_round_3/val_res.pkl', 'N=5_student_round_4/val_res.pkl']

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

        self.validation_dataloaders = Multitask_DatasetDataLoader(
            train_mode = 'Validation', num_threads = args.n_threads_test, dataset_names=args.dataset_names,
            tasks = args.tasks, batch_size = 1, seq_len = 64, fps = 30, # validation set always sample by 30 fps
            transform = test_transforms(args.image_size))
        self.validation_dataloaders = self.validation_dataloaders.load_multitask_val_test_data()
        print("Validation sets loaded")
        #self.evalute_ours()
        #self.evalute_TS()
        self.evaluate_Monte_Carol_Dropout()
    def evaluate_Monte_Carol_Dropout(self):
        save_file = 'mc_dropout_res.pkl'
        tasks = ['AU', 'EXPR', 'VA']
        if not os.path.exists(save_file):
            track_val_preds, track_val_labels = self._validate_tasks_metrics_MC(tasks=tasks)
            pickle.dump([track_val_preds, track_val_labels], open(save_file, 'wb'))
        else:
            track_val_preds, track_val_labels = pickle.load(open(save_file, 'rb'))
        
        for task in tasks:
            preds_total = []
            for i_model in track_val_preds[task].keys():
                preds, labels = track_val_preds[task][i_model], track_val_labels[task][i_model]
                preds_total.append(preds)

            # evaluate MC dropout for task (average the output probabilities)
            total_probas = []
            for preds in preds_total:
                p = self.logits_2_probas(preds, task)
                total_probas.append(p)
            probas = np.mean(np.stack(total_probas, axis=0), axis=0)
            NLLs = self.get_NLL(probas, labels, task)
            output = "MC dropout "+task +" "
            if task=='AU':
                num_aus = NLLs.shape[-1]
                NLLs = NLLs.mean(0) # num_samples, num_aus
                means = []
                for i in range(num_aus):
                    mean = NLLs[i]
                    output +="{:.3f} & ".format(mean)
                    means.append(mean)
                output +='{:.3f}'.format(np.mean(means))
                
            elif task == 'EXPR':
                output += '{:.3f}'.format(NLLs.mean())
            else:
                rmse = self.get_RMSE(probas, labels, task)
                output += 'V {:.3f} A {:.3f} '.format(rmse[0], rmse[1])
            print(output)
        # eval_items, eval_res = metric_func(estimates.reshape(B*N, -1).squeeze(), copy(labels.reshape(B*N, -1).squeeze()))
        # print("Ensemble {} eval res: {}, eval 0: {}, eval 1 {}".format(task, eval_res,
        #                     eval_items[0], eval_items[1]))

    def _validate_tasks_metrics_MC(self, n_forwards = 10, tasks = ['AU', 'EXPR', 'VA']):
        eval_per_task = {}
        hiddens = dict([(i_model, None) for i_model in range(n_forwards)])
        video_name = None
        assert len(self._models) == 1
        model = self._models[0]
        model._model.tmodels.training = True # turn on dropout of the temporal models, not the dropout of backbone
        record_metrics_single = {}
        track_val_preds = {}
        track_val_labels = {}
        for task in tasks:
            track_val_preds[task] = {}
            track_val_labels[task] = {}
            data_loader = self.validation_dataloaders[task]
            for i_val_batch, val_batch in tqdm(enumerate(data_loader), total = len(data_loader)):
                wrapped_v_batch = {task: val_batch}
                if video_name is None:
                    video_name = val_batch['video'][0]
                else:
                    if video_name != val_batch['video'][0]:
                        hiddens = dict([(i_model, None) for i_model in range(n_forwards)])

                for i_model in range(n_forwards): # n_forwards forward passes
                    model.set_input(wrapped_v_batch, input_tasks = [task])
                    outputs, errors, hiddens[i_model] = model.forward(return_estimates=False, 
                        input_tasks = [task], FA_teacher=FA_teacher, hiddens = hiddens[i_model],
                        MC_dropout=True)
                    
                    if i_model not in track_val_preds[task].keys():
                        track_val_preds[task][i_model] = []
                        track_val_labels[task][i_model] = []
                    B, N = outputs[task][task].shape[:2]
                    track_val_preds[task][i_model].append(outputs[task][task].reshape(B*N,-1))
                    track_val_labels[task][i_model].append(wrapped_v_batch[task]['label'].reshape(B*N,-1).squeeze(-1))

                # if i_val_batch>100:
                #     break
            for i_model in track_val_preds[task].keys():
                track_val_preds[task][i_model] = np.concatenate(track_val_preds[task][i_model], axis=0)
                track_val_labels[task][i_model] = np.concatenate(track_val_labels[task][i_model], axis=0)
        return track_val_preds, track_val_labels


    def evalute_TS(self):
        res_file =  res_files[0] # load the teacher predictions and apply temperature scaling
        optimal_Ts = {}
        NLLs_TS = {}
        [_, _], [track_val_preds, track_val_labels] = pickle.load(open(res_file, 'rb'))
        for task in ['AU', 'EXPR']:
            preds_dict = track_val_preds[task]
            labels_dict = track_val_labels[task]
            probas_total = []
            optimal_Ts[task] = {}
            NLLs_TS[task] = {}
            for i_model in tqdm(preds_dict.keys()):
                preds, labels = preds_dict[i_model], labels_dict[i_model]
                B, N = preds.shape[:2]
                preds, labels = preds.reshape(B*N, -1).squeeze(), labels.reshape(B*N, -1).squeeze()

                # create a random mask for i th model
                N = len(preds)
                mask = np.array([True]*N)
                indexes = np.random.choice(np.arange(N), size = N//2, replace=False)
                mask[indexes] = False

                # optimize temperature on a randomly selected subset using mask
                T = self.optimize_temperature(copy(preds[mask]), copy(labels[mask]), task)
                optimal_Ts[task][i_model] = T

                #nll compute
                probas_with_temperature = self.logits_2_probas(copy(preds[~mask]), task, T = T)
                NLLs_TS[task][i_model] = self.get_NLL(probas_with_temperature, copy(labels[~mask]), task)
            output = "TS: "+task +" "

            NLLs = np.stack([NLLs_TS[task][i_model] for i_model in NLLs_TS[task].keys()], axis=0)
            if task == 'AU':
                num_aus = NLLs.shape[-1]
                NLLs = NLLs.mean(1) # N_models, num_samples, num_aus
                means = []
                for i in range(num_aus):
                     mean, std = np.mean(NLLs[:, i]), np.std(NLLs[:, i])
                     output +="{:.3f} & ".format(mean)
                     means.append(mean)
                output +='{:.3f}'.format(np.mean(means))
                print(output)
            elif task =='EXPR':
                print(output+"{:.3f}".format(NLLs.mean()))
        return NLLs_TS
    def evalute_ours(self):
        for res_file in res_files:
            assert os.path.exists(res_file)
            [single_model_metrics, ensemble_metrics], [track_val_preds, track_val_labels] = pickle.load(open(res_file, 'rb'))
            ours_NLLs, ours_RMSEs = self._validate_NLL_and_RMSE(track_val_preds, track_val_labels)
            print('res file: {}'.format(res_file))
            
            for task in ['AU', 'EXPR', 'VA']:
                output = task +':'
                NLLs = np.stack([ours_NLLs[task][i_model] for i_model in ours_NLLs[task].keys()], axis=0)
                if task == 'AU':
                    C = NLLs.shape[-1]
                    NLLs = NLLs.mean(1) # N_models, num_samples, C
                    means = []
                    for i in range(C):
                         mean, std = np.mean(NLLs[:, i]), np.std(NLLs[:, i])
                         output +="{:.3f} & ".format(mean)
                         means.append(mean)
                    output +='{:.3f}'.format(np.mean(means))
                    print(output)
                elif task == 'EXPR':
                    NLLs = NLLs.mean(1) # N_models, num_samples
                    output +='{:.3f}'.format(np.mean(NLLs))
                    print(output)
                elif task == 'VA':
                    RMSEs = np.stack([ours_RMSEs[task][i_model] for i_model in ours_RMSEs[task].keys()], axis=0)
                    RMSEs = RMSEs.mean(0)
                    print("V: {:.3f}, A: {:.3f}".format(RMSEs[0], RMSEs[1]))
    def _validate_NLL_and_RMSE(self, track_val_preds, track_val_labels, tasks=['AU', 'EXPR', 'VA']):
        optimal_Ts = {}
        NLLs = {}
        RMSEs = {}
        for task in tasks:
            preds_dict = track_val_preds[task]
            labels_dict = track_val_labels[task]
            probas_total = []
            optimal_Ts[task] = {}
            NLLs[task] = {}
            RMSEs[task] = {}
            for i_model in tqdm(preds_dict.keys()):
                preds, labels = preds_dict[i_model], labels_dict[i_model]
                B, N = preds.shape[:2]
                preds, labels = preds.reshape(B*N, -1).squeeze(), labels.reshape(B*N, -1).squeeze()
                probas = self.logits_2_probas(preds, task)
                probas_total.append(probas)
                NLLs[task][i_model] = self.get_NLL(probas, labels, task)
                if task =='VA':
                    RMSEs[task][i_model] = self.get_RMSE(probas, labels, task)
            # # regression nll
            # if task == 'VA':
            #     edges = np.linspace(-1, 1, 20)
            #     output = 'regression NLL:'
            #     for i in range(2):
            #       preds_total = [(proba[:, i*20:(i+1)*20]*edges).sum(-1) for proba in probas_total]
            #       preds_mean = np.stack(preds_total, axis=0).mean(0)
            #       preds_std = np.std(np.stack(preds_total, axis=0), axis=0)
            #       nll= (labels[:, i] - preds_mean)**2/(2*preds_std**2) + 0.5*np.log(preds_std**2) + 0.5*np.log(2*np.pi)
            #       nll = nll.mean()
            #       if i==0:
            #         output+='V {:.3f} '.format(nll)
            #       else:
            #         output+='A {:.3f}'.format(nll)
            #    print(output)

        return NLLs, RMSEs

    def optimize_temperature(self, logits, labels, task):
        if task == 'EXPR':
            temperature = TemperatureScaling()
            temperature.fit(self.logits_2_probas(logits, task), labels)
            t = 1/temperature.temperature[0]
        elif task == 'AU':
            ts = []
            for i in range(labels.shape[1]):
                temperature = TemperatureScaling()
                temperature.fit(self.logits_2_probas(logits[:, i], task), labels[:, i])
                t = 1/temperature.temperature[0]
                ts.append(t)
            t = np.array(ts)
        elif task == 'VA':
            ts = []
            edges = np.linspace(-1, 1, 20)
            for i in range(labels.shape[1]):
                temperature = TemperatureScaling()
                l = labels[:, i]
                dl = np.digitize(l, edges,right=True)
                dl[dl==20] = 19
                temperature.fit(self.logits_2_probas(logits[:, i*20:(i+1)*20], task), dl)
                t = 1/temperature.temperature[0]
                ts.append(t)
            t = np.array(ts)
        return t
    def to_onehot_labels(self, labels, num_classes = 7, task= 'EXPR'):
        if task == 'EXPR':
            onehot = np.zeros((len(labels),num_classes))
            onehot[np.arange(len(labels)),labels] = 1
        elif task == 'AU':
            onehot = labels
        elif task == 'VA':
            edges = np.linspace(-1, 1 , 21)
            if num_classes==40:
                v, a = labels[::, 0], labels[::, 1]
                dig_v = np.digitize(v, edges,right=True)
                dig_v[dig_v == 20] = 20 - 1
                dig_a = np.digitize(a, edges,right=True)
                dig_a[dig_a == 20] = 20 -1
                num_classes = 20
                onehot_v = np.zeros((len(labels), num_classes))
                onehot_v[np.arange(len(labels), dig_v)] = 1
                onehot_a= np.zeros((len(labels), num_classes))
                onehot_a[np.arange(len(labels), dig_a)] = 1
                labels = np.stack([onehot_v, onehot_a], axis=-1)
                onehot = labels
            elif num_classes == 20:
                dig = np.digitize(labels, edges,right=True)
                dig[dig==20] =19
                onehot = np.zeros((len(labels), num_classes))
                onehot[np.arange(len(labels)), dig] = 1
        return onehot

    def logits_2_probas(self, preds, task, T=None):
        if task == 'EXPR':
            if T is None:
                T = 1
            preds = preds/T
            p = softmax(preds, axis=-1)
        elif task =='AU':
            if T is None:
                T = 1
            preds = preds/T
            p = sigmoid(preds)
        elif task =='VA':
            if preds.shape[-1]==20:
                if T is None:
                    T = 1
                p = softmax(preds/T, axis=-1)
            elif preds.shape[-1]==40:  
                if T is None:
                    T = [1,1]
                assert len(T) == 2
                p = [softmax(preds[..., :20]/T[0], axis=-1), softmax(preds[..., 20:]/T[1], axis=-1)]
                p = np.concatenate(p, axis=-1)
        return p
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

    def get_NLL(self, probas, labels, task):
        if task == 'AU':
            nll = -(labels*np.log(probas + EPS) + (1-labels)*np.log(1-probas + EPS))
            return nll
        elif task == 'EXPR':
            nll = - np.log(probas[np.arange(len(labels)), labels] + EPS)
            return nll
        elif task == 'VA':
            edges = np.linspace(-1, 1, 20)
            nlls = []
            for i in range(2):
                l = labels[:, i]
                dl = np.digitize(l, edges,right=True)
                dl[dl==20] = 19
                # give the adjacent three bins the same probability
                onehot_labels = np.zeros((len(l), 20))
                onehot_labels[np.arange(len(l)), dl] = 1
                # new_dl = dl+1
                # new_dl[new_dl==20] = 19
                # onehot_labels[np.arange(len(l)), new_dl] = 1/3
                # new_dl = dl-1
                # new_dl[new_dl==-1] = 0
                # onehot_labels[np.arange(len(l)), new_dl] = 1/3
                # onehot_labels = onehot_labels/onehot_labels.sum(-1).reshape(len(dl), -1)
                p = probas[:, i*20 :(i+1)*20]
                nll = (- onehot_labels*np.log(p +EPS)).sum(-1)
                nlls.append(np.array(nll))
            return np.stack(nlls, axis=-1)
    def get_RMSE(self, probas, labels, task):
        assert task == 'VA'
        edges = np.linspace(-1, 1, 20)
        rmses = []
        for i in range(2):
            l = labels[:, i]
            p = probas[:, i*20 :(i+1)*20]
            p = (p*edges).sum(-1)
            rmse = np.sqrt((( p - l)**2).mean())
            rmses.append(rmse)
        return np.stack(rmses, axis=-1)

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
