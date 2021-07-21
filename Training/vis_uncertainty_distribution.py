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
from utils.misc import cal_uncertainty
import matplotlib
import matplotlib.pyplot as plt
font = {
    'size'   : 28}
matplotlib.rc('font', **font)
#plt.rcParams.update({'figure.autolayout': True})
plt.rcParams["font.family"] = "serif"
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
res_files = ['N=5_student_round_3/val_res.pkl']
OOD_res_files = ['OOD/val_res.pkl']
imgs_dir = 'uncertainty_performances'
if not os.path.exists(imgs_dir):
    os.makedirs(imgs_dir)
PRESET_VARS = PATH()
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

        self.validation_dataloaders = Multitask_DatasetDataLoader(
            train_mode = 'Validation', num_threads = args.n_threads_test, dataset_names=args.dataset_names,
            tasks = args.tasks, batch_size = 1, seq_len = 64, fps = 30, # validation set always sample by 30 fps
            transform = test_transforms(args.image_size))
        self.validation_dataloaders = self.validation_dataloaders.load_multitask_val_test_data()
        print("Validation sets loaded")
        res_file = res_files[0]
        assert os.path.exists(res_file)
        [single_model_metrics, ensemble_metrics], [track_val_preds, track_val_labels] = pickle.load(open(res_file, 'rb'))
        data_dict = self.evalute_on_val(track_val_preds, track_val_labels)
        OOD_res_file = OOD_res_files[0]
        track_val_preds, track_val_labels = pickle.load(open(OOD_res_file, 'rb'))
        ood_data_dict = self.evalute_on_val(track_val_preds, track_val_labels)
        self.vis_data_dict(data_dict, name = 'in_domain')
        self.vis_data_dict(ood_data_dict, name = 'OOD')


    def vis_data_dict(self, data_dict, name='in_domain'):
        for task in data_dict.keys():
            proba, labels,  uncertainty = data_dict[task]
            uncertainty_ids = [0, 1]
            for u_id in uncertainty_ids:
                if task == 'EXPR' or task =='AU':
                    #self.vis_AU_or_EXPR(proba, uncertainty, task, u_id, name)
                    pass
                else:
                    self.vis_VA(proba, uncertainty, task, u_id, name)
                    pass

    def vis_VA(self, proba, uncertainty, task='VA',u_id= 0, filename=""):
        uncertainty = uncertainty[:,:,  u_id]
        C= 2
        edges = np.linspace(-1, 1, 20)
        preds = [(proba[:, i*20: (i+1)*20]*edges).sum(-1) for i in range(2)]
        fig, axes = plt.subplots(1, C, figsize=(16, 4))
        for i in range(C):
            un_score = uncertainty[:, i]
            name = 'Valence' if i==0 else 'Arousal'
            # downsample mask
            mask = np.array([0]*len(un_score)).astype(np.bool)
            mask[np.arange(len(un_score))[::100]] = 1
            mask[preds[i]<-0.5] = 1
            axes[i].scatter(preds[i][mask], un_score[mask])
            axes[i].set_xlabel(name)
            axes[i].set_xlim([-1, 1])
            axes[i].set_xticks(np.linspace(-1, 1, 5))
            ylabel = "Aleatoric\nuncertainty" if u_id ==0 else "Epistemic\nuncertainty"
            #axes[i].set_ylabel(ylabel)
            axes[i].set_yticks([0, 1])
            axes[i].grid()
        fig.subplots_adjust(bottom=0.2)
        save_path = os.path.join(imgs_dir, "{}_{}_{}.pdf".format(task, ylabel.split('\n')[0], filename))
        plt.savefig(save_path, bbox_inches='tight',)
        plt.show()

    def vis_AU_or_EXPR(self, proba, uncertainty, task = 'EXPR', u_id=0, filename=""):
        categories = PRESET_VARS.Aff_wild2.categories[task]
        if len(uncertainty.shape) == 2:
            uncertainty = uncertainty[:, u_id]
            labels = proba.argmax(-1) # predictions
            categories = ['N', 'A', 'D', 'F', 'H', 'Sa', 'Su']
        elif len(uncertainty.shape) == 3:
            uncertainty = uncertainty[:,:,  u_id]
            labels = (proba>0.5).astype(np.int) # predictions
            categories = [c.split("AU")[-1] for c in categories]
        C = 12 if task =='AU' else 7 
        
        fig, axes = plt.subplots(1, C, figsize=(10, 3))
        nbins = 21
        hists = []
        max_ratio = 0

        for i in range(C):
            if task == 'AU':
                un_score = uncertainty[:, i]
            else:
                mask = labels==i
                un_score = uncertainty[mask]
            hist, _ = np.histogram(un_score, bins=np.linspace(0, 1, nbins))
            hist = hist/hist.sum()
            max_ratio = max(max_ratio, hist.max())
            hists.append(hist)
        
        for i in range(C):
            hist = hists[i]
            hist= np.stack([hist]*4, axis=-1)
            img = axes[i].imshow(hist[::-1, :], vmin=0, vmax=np.around(max_ratio, decimals=2))
            axes[i].get_xaxis().set_ticks([])
            axes[i].set_xlabel(categories[i])
            
            if i==0:
                axes[i].set_yticks([0, nbins-2])
                axes[i].set_yticklabels([1, 0])
                ylabel = "Aleatoric\nuncertainty" if u_id ==0 else "Epistemic\nuncertainty"
                #axes[i].set_ylabel(ylabel)
            else:
                axes[i].set_yticks([])
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        cbar = fig.colorbar(img, cax=cbar_ax, )
        save_path = os.path.join(imgs_dir, "{}_{}_{}.pdf".format(task, ylabel.split('\n')[0], filename))
        plt.savefig(save_path, bbox_inches='tight',)
        plt.show()

    def evalute_on_val(self,track_val_preds, track_val_labels, tasks = ['AU', 'EXPR', 'VA']):
        output_dict = {}
        for task in tasks:
            N_models = len(track_val_preds[task].keys())
            preds = np.stack([track_val_preds[task][i_model] for i_model in track_val_preds[task].keys()], axis=0)
            if task in track_val_labels.keys():
                labels = track_val_labels[task][0]
            else:
                labels = None
            total_probas = []
            for i_model in range(N_models):
                preds_i_model = track_val_preds[task][i_model]
                B, N = preds_i_model.shape[:2]
                preds_i_model = preds_i_model.reshape(B*N, -1)
                if task in track_val_labels.keys():
                    labels = labels.reshape(B*N, -1).squeeze()
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
            o=  [mean_probas, labels,  np.stack([alea_uncertainty, epi_uncertainty], axis=-1)]
            output_dict[task] = o
        return output_dict
            
if __name__=="__main__":
    Validator()
