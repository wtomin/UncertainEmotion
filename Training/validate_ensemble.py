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
from netcal.scaling import TemperatureScaling
from netcal.metrics import ECE 
from sklearn.metrics import roc_curve
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
        # from thop import profile
        # input = torch.randn(1, 1, 3, 112, 112).cuda()
        # macs, params = profile(self._models[0]._model, inputs=(input, ))
        # validation dataset 
        self.validation_dataloaders = Multitask_DatasetDataLoader(
            train_mode = 'Validation', num_threads = args.n_threads_test, dataset_names=args.dataset_names,
            tasks = args.tasks, batch_size = 1, seq_len = 64, fps = 30, # validation set always sample by 30 fps
            transform = test_transforms(args.image_size))
        self.validation_dataloaders = self.validation_dataloaders.load_multitask_val_test_data()
        print("Validation sets")
        
        for task in args.tasks:
            if task in self.validation_dataloaders.keys():
                data_loader = self.validation_dataloaders[task]
                print("{}: {} images".format(task, len(data_loader)*args.batch_size * args.seq_len))
        save_file = 'N=5/val_res.pkl'
        if not os.path.exists(save_file):
            [single_model_metrics, ensemble_metrics], [track_val_preds, track_val_labels] = self._validate_tasks_metrics()
            pickle.dump([[single_model_metrics, ensemble_metrics], [track_val_preds, track_val_labels]], open(save_file, 'wb'))
        else:
            [single_model_metrics, ensemble_metrics], [track_val_preds, track_val_labels] = pickle.load(open(save_file, 'rb'))
        NLLs, NLLs_TS, ECEs, ECEs_TS = self._validate_uncertainty_metrics(track_val_preds, track_val_labels)
        self._visualize_uncertainty_metrics(NLLs, NLLs_TS, ECEs, ECEs_TS)
        self._visualize_task_metrics(single_model_metrics, ensemble_metrics)
    def _visualize_uncertainty_metrics(self, NLLs, NLLs_TS, ECEs, ECEs_TS):
        fig, axes = plt.subplots(len(NLLs.keys()), 2)
        for i, task in enumerate(NLLs.keys()):
            data_nll = [ [NLLs[task][i_model] for i_model in NLLs_TS[task].keys()],
                         [NLLs_TS[task][i_model] for i_model in NLLs_TS[task].keys()],
                         [NLLs[task]['ens']]]
            data_ece = [ [ECEs[task][i_model] for i_model in ECEs_TS[task].keys()],
                         [ECEs_TS[task][i_model] for i_model in ECEs_TS[task].keys()],
                         [ECEs[task]['ens']]]
            axes[i][0].boxplot(data_nll, positions = np.arange(1, len(data_nll)+1), widths=0.25)
            axes[i][1].boxplot(data_ece, positions = np.arange(1, len(data_ece)+1), widths=0.25)
            axes[i][0].set_title("{}:{}".format(task, "NLL"))
            axes[i][1].set_title("{}:{}".format(task, "ECE"))
            axes[i][0].set_xticks(np.arange(1, len(data_nll)+1))
            axes[i][1].set_xticks(np.arange(1, len(data_nll)+1))
            axes[i][0].set_xticklabels(['Vanilla', 'TemperatureScaling', 'Deep Ensemble'])
            axes[i][1].set_xticklabels(['Vanilla', 'TemperatureScaling', 'Deep Ensemble'])

        plt.show()

    def _validate_uncertainty_metrics(self, track_val_preds, track_val_labels):
        tasks = list(track_val_labels.keys())
        if 'VA' in tasks:
            tasks.remove('VA') # do not compute NLL or ECE for valence and arousal
        optimal_Ts = {}
        NLLs_TS = {}
        NLLs = {}
        ECEs_TS = {}
        ECEs = {}
        for task in tasks:
            preds_dict = track_val_preds[task]
            labels_dict = track_val_labels[task]
            probas_total = []
            optimal_Ts[task] = {}
            NLLs[task] = {}
            NLLs_TS[task] = {}
            ECEs_TS[task] = {}
            ECEs[task] = {}
            for i_model in preds_dict.keys():
                preds, labels = preds_dict[i_model], labels_dict[i_model]
                B, N = preds.shape[:2]
                preds, labels = preds.reshape(B*N, -1).squeeze(), labels.reshape(B*N, -1).squeeze()
                probas = self.logits_2_probas(preds, task)
                probas_total.append(probas)
                N = len(preds)
                mask = np.array([True]*N)
                indexes = np.random.choice(np.arange(N), size = N//2, replace=False)
                mask[indexes] = False
                T = self.optimize_temperature(copy(preds[mask]), copy(labels[mask]), task)
                optimal_Ts[task][i_model] = T
                NLLs_TS[task][i_model] = self.get_NLL(
                    self.logits_2_probas(copy(preds[~mask]), task, T = T), 
                    copy(labels[~mask]), task)
                NLLs[task][i_model] = self.get_NLL(
                    probas, labels, task)
                ECEs_TS[task][i_model] = self.get_ECE(
                    self.logits_2_probas(copy(preds[~mask]), task, T = T), 
                    copy(labels[~mask]))
                ECEs[task][i_model] = self.get_ECE(
                    copy(probas), 
                    copy(labels))
            probas_total = np.stack(probas_total, axis=0).mean(0)
            NLLs[task]['ens'] = self.get_NLL(probas_total, labels, task)
            ECEs[task]['ens'] = self.get_ECE(probas_total, labels)
        for task in optimal_Ts.keys():
            output = "Task: {}".format(task)
            output += " ".join(['{}: {}'.format(i_model, optimal_Ts[task][i_model]) for i_model in optimal_Ts[task].keys()])
            print(output)
        return NLLs, NLLs_TS, ECEs, ECEs_TS

    def get_ECE(self, probas, labels, nbins= 10):
        ece = ECE(bins = nbins)
        if len(labels.shape) == 1:
            ece_score = ece.measure(probas, labels)
        else:
            N = labels.shape[1]
            ece_score = []
            for i in range(N):
                p, l = probas[:, i], labels[:, i]
                ece_score.append(ece.measure(p, l))
            ece_score = np.mean(ece_score)
        return ece_score

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
        return t

    def _visualize_task_metrics(self, single_model_metrics, ensemble_metrics):
        y1, y1_err, y2 = [], [], []
        tasks = list(single_model_metrics.keys())
        if 'VA' in tasks and 'Valence' in tasks:
            tasks.remove('VA')
        for task in tasks:
            metrics = single_model_metrics[task]
            mu, var = np.mean(metrics), np.std(metrics)
            y1.append(mu)
            y1_err.append(var)
            metric = ensemble_metrics[task]
            y2.append(metric)
        X = np.arange(len(tasks)) + 1
        plt.subplots(1, 1)
        plt.bar(X, y1, yerr=y1_err, width=0.3, facecolor = 'lightskyblue', edgecolor = 'white', label='single')
        plt.bar(X+0.3, y2, width=0.3, facecolor = 'yellowgreen', edgecolor = 'white', label='ensemble')
        plt.legend(loc="upper left")
        task_labels = [task+" CCC" if task =='Valence' or task =='Arousal' else task +" Metric" for task in tasks  ]
        plt.xticks(X, task_labels)
        plt.show()

    def _validate_tasks_metrics(self):
        eval_per_task = {}
        tasks = copy(args.tasks)
        if 'FA' in tasks:
            tasks.remove('FA')
        hiddens = None
        video_name = None
        optimal_Ts = {}
        FA_metrics = {}
        record_metrics_single = {}
        record_metrics_ensemble = {}
        FA_preds = {}
        FA_labels = {}
        track_val_preds = {}
        track_val_labels = {}
        AU_optimal_thresholds = []
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
                        hiddens = None

                for i_model, model in enumerate(self._models):
                    model.set_input(wrapped_v_batch, input_tasks = [task])
                    outputs, errors, hiddens = model.forward(return_estimates=False, 
                        input_tasks = [task], FA_teacher=FA_teacher, hiddens = hiddens)
                    
                    if i_model not in track_val_preds[task].keys():
                        track_val_preds[task][i_model] = []
                        track_val_labels[task][i_model] = []
                        FA_preds[i_model] = []
                        FA_labels[i_model] = []
                        FA_metrics[i_model] = []
                    track_val_preds[task][i_model].append(outputs[task][task])
                    track_val_labels[task][i_model].append(wrapped_v_batch[task]['label'])
                    FA_metrics[i_model].append(errors['loss_FA'])
                    with torch.no_grad():
                        FA_labels[i_model].append(FA_teacher(wrapped_v_batch[task]['image'].squeeze(0).cuda()).unsqueeze(0).cpu().numpy())
                        FA_preds[i_model].append(outputs[task]['FA'])
                # if i_val_batch>100:
                #     break
            for i_model in track_val_preds[task].keys():
                track_val_preds[task][i_model] = np.concatenate(track_val_preds[task][i_model], axis=0)
                track_val_labels[task][i_model] = np.concatenate(track_val_labels[task][i_model], axis=0)

            # evalute each single model for task 
            preds_total = []
            record_metrics_single[task] = []
            for i_model in track_val_preds[task].keys():
                preds, labels = track_val_preds[task][i_model], track_val_labels[task][i_model]
                B, N = preds.shape[:2]
                metric_func = self._models[i_model].get_metrics_per_task()[task]
                if task != 'AU':
                    estimates = self._models[i_model]._format_estimates({task: torch.FloatTensor(track_val_preds[task][i_model])})
                else:
                    estimates, thresholds = self.AU_preds_to_estimates(preds, labels)
                    AU_optimal_thresholds.append(thresholds)
                eval_items, eval_res = metric_func(estimates[task].reshape(B*N, -1).squeeze(), copy(labels.reshape(B*N, -1).squeeze()))
                print("Model {} {} eval res: {}, eval 0: {}, eval 1 {}".format(i_model, task, eval_res,
                    eval_items[0], eval_items[1]))
                if task !='VA':
                    record_metrics_single[task].append(eval_res)
                else:
                    if 'Valence' not in record_metrics_single.keys():
                        record_metrics_single['Valence'] = []
                    if 'Arousal' not in record_metrics_single.keys():
                        record_metrics_single['Arousal'] = []
                    record_metrics_single['Valence'].append(eval_items[0])
                    record_metrics_single['Arousal'].append(eval_items[1])
                preds_total.append(preds)

            # evaluate ensemble model for task (average the output probabilities)
            if task == 'AU':
                for thresholds in AU_optimal_thresholds:
                    print("AU thresholds:{}".format(thresholds))
                AU_optimal_thresholds = np.stack(AU_optimal_thresholds, axis=0).mean(0)
            probas = []
            for preds in preds_total:
                p = self.logits_2_probas(preds, task)
                probas.append(p)
            probas = np.mean(np.stack(probas, axis=0), axis=0)
            estimates = self.probas_2_estimates(probas, task, AU_optimal_thresholds if task=='AU' else None)
            B, N = estimates.shape[:2]
            eval_items, eval_res = metric_func(estimates.reshape(B*N, -1).squeeze(), copy(labels.reshape(B*N, -1).squeeze()))
            print("Ensemble {} eval res: {}, eval 0: {}, eval 1 {}".format(task, eval_res,
                                eval_items[0], eval_items[1]))
            if task !='VA':
                record_metrics_ensemble[task] = eval_res
            else:
                record_metrics_ensemble['Valence'] = eval_items[0]
                record_metrics_ensemble['Arousal'] = eval_items[1]

        for i_model in FA_metrics.keys():
            FA_metrics[i_model] = np.mean(FA_metrics[i_model])
            FA_labels[i_model] = np.concatenate(FA_labels[i_model], axis=0)
            FA_preds[i_model] = np.concatenate(FA_preds[i_model], axis=0)

        record_metrics_single['FA'] = [FA_metrics[i_model] for i_model in FA_metrics.keys()]
        preds = np.stack([FA_preds[i_model] for i_model in FA_preds.keys()], axis=0).mean(0)
        l1_loss = np.abs(preds - FA_labels[0]).mean()
        record_metrics_ensemble['FA'] = l1_loss
        print("FA single models:{}".format(record_metrics_single['FA']))
        print("FA ensemble models:{}".format(record_metrics_ensemble['FA']))
        return [record_metrics_single, record_metrics_ensemble], [track_val_preds, track_val_labels]

    def AU_preds_to_estimates(self, preds, labels):
        probas = self.logits_2_probas(preds, task = 'AU')
        C = probas.shape[-1]
        optimal_thresholds = []
        for i_c in range(C):
            p = probas[..., i_c]
            l = labels[..., i_c]
            fpr, tpr , thresholds = roc_curve(l.reshape(-1,), p.reshape(-1,))
            J = tpr -fpr
            i = np.argmax(J)
            optimal_thresholds.append(thresholds[i])
        optimal_thresholds = np.array(optimal_thresholds)
        estimates = (probas>optimal_thresholds).astype(np.int)
        return {'AU': estimates}, optimal_thresholds

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
    def probas_2_estimates(self, probas, task, AU_thresholds= None):
        if task == 'EXPR':
            est = probas.argmax(axis=-1).astype(np.int)
        elif task =='AU':
            if AU_thresholds is None:
                est = (probas > 0.5).astype(np.int)
            else:
                est = (probas > AU_thresholds).astype(np.int)
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






            
        
