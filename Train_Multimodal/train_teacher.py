import time
import gc
from options.train_options import TrainOptions
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
from utils.transforms import train_transforms, test_transforms
from utils.options import prepare_arguments
from torch.utils.tensorboard import SummaryWriter
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
args = TrainOptions().parse()

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

class Trainer:
    def __init__(self):
        PRESET_VARS = PATH()
        self._model = ModelsFactory.get_by_name(args,
            is_train= True,
            dropout = 0.5,
            pretrained=True)
        model = self._model._model
        print("number of parameters: {}".format(sum(p.numel() for p in model.parameters())))
        self.training_dataloaders = Multitask_DatasetDataLoader(train_mode = 'Train', 
            num_threads = args.n_threads_train, dataset_names=args.dataset_names,
            tasks = args.tasks, batch_size = args.batch_size, seq_len = args.seq_len, fps = args.fps, window_size=args.window_size, sr = args.sr,
            transform = train_transforms(args.image_size))
        self.training_dataloaders = self.training_dataloaders.load_multitask_train_data()

        self.validation_dataloaders = Multitask_DatasetDataLoader(
            train_mode = 'Validation', num_threads = args.n_threads_test, dataset_names=args.dataset_names,
            tasks = args.tasks, batch_size = 1, seq_len = 64, fps = 30, window_size=args.window_size, sr = args.sr,
            transform = test_transforms(args.image_size))
        self.validation_dataloaders = self.validation_dataloaders.load_multitask_val_test_data()
        
        print("Traning tasks {} on datasets: {}".format(args.tasks, args.dataset_names))
        actual_bs = args.batch_size* len(args.dataset_names)
        print("The actual batch size is {}*{}={}".format(args.batch_size, len(args.dataset_names), actual_bs))
        print("Training sets: {} images ({} images per task)".format(len(self.training_dataloaders) * actual_bs * args.seq_len, 
            len(self.training_dataloaders)* args.batch_size* args.seq_len))
        print("Validation sets")
        
        for task in args.tasks:
            if task in self.validation_dataloaders.keys():
                data_loader = self.validation_dataloaders[task]
                print("{}: {} images".format(task, len(data_loader)*args.batch_size * args.seq_len))
        self.writer = SummaryWriter(log_dir = os.path.join(args.loggings_dir, args.name), 
                                 filename_suffix = args.name)
        self._train()
        self.writer.close()
    def _train(self):
        self._total_steps = args.load_epoch * len(self.training_dataloaders) 
        self._last_save_time = time.time()
        self._last_print_time = time.time()
        self._current_val_acc = dict([(t, 0.) for t in args.tasks if t !='FA'] + [('FA', 1000)])
        self._no_improve_n_epochs = dict([(t, 0) for t in args.tasks])
        for t in args.tasks:
            self.writer.add_scalar("Lambdas/{}".format(t), self._model.lambdas_per_task[t], 0)
        if args.load_epoch !=0: # lr scheduler adjust
            for _ in range(args.load_epoch):
                if args.lr_policy == 'step':
                    self._model._LR_scheduler.step()
                elif args.lr_policy == 'cosine':
                    for _ in range(len(self.training_dataloaders)):
                        self._model._LR_scheduler.step()
        for i_epoch in range(args.load_epoch + 1, args.nepochs + 1):
            epoch_start_time = time.time()
            self._model.get_current_LR()
            # train epoch
            self._train_epoch(i_epoch)
            self.training_dataloaders.reset()
            if args.lr_policy == 'step':
                self._model._LR_scheduler.step()
            val_dict = self._validate(i_epoch)
            gc.collect()
            val_acc = sum([val_dict[t] for t in args.tasks if (t !='FA') and (t!='VAD')])
            cur_val_acc = sum([self._current_val_acc[t] for t in args.tasks if (t !='FA') and (t!='VAD')])
            if val_acc > cur_val_acc:
                print("validation acc improved, from {:.4f} to {:.4f}".format(cur_val_acc, val_acc))

            for t in args.tasks:
                metric = val_dict[t]
                if (t!='FA' and metric> self._current_val_acc[t]) or (t =='FA' and metric < self._current_val_acc[t]):
                    self._current_val_acc[t] = metric
                    self._no_improve_n_epochs[t] = 0
                else:
                    self._no_improve_n_epochs[t] +=1

            self._model.update_lambda(self._no_improve_n_epochs)
            for t in args.tasks:
                self.writer.add_scalar("Lambdas/{}".format(t), self._model.lambdas_per_task[t], i_epoch)
            self.writer.add_scalar("Val_metric/total", val_acc, i_epoch)

            self._model.save(i_epoch) # save every epoch model

            # print epoch info
            time_epoch = time.time() - epoch_start_time
            print('End of epoch %d / %d \t Time Taken: %d sec (%d min or %d h)' %
                  (i_epoch, args.nepochs , time_epoch,
                   time_epoch / 60, time_epoch / 3600))

    def _train_epoch(self, i_epoch):

        self._model.set_train()
        for i_train_batch, train_batch in enumerate(self.training_dataloaders):
            iter_start_time = time.time()
            # display flags

            do_print_terminal = time.time() - self._last_print_time > args.print_freq_s 
            do_save = time.time() - self._last_save_time > args.save_freq_s
            # train model
            self._model.set_input(train_batch)

            self._model.optimize_parameters(FA_teacher=FA_teacher, VAD_teacher =VAD_teacher)

            # update epoch info
            self._total_steps += 1

            if args.lr_policy == 'cosine':
                self._model._LR_scheduler.step()

            # display terminal
            if do_print_terminal:
                self._display_terminal(iter_start_time, i_epoch, i_train_batch, len(self.training_dataloaders))
                self._last_print_time = time.time()
            if do_save:
                for key in self._model.loss_dict.keys():
                    self.writer.add_scalar('Train/{}'.format(key), self._model.loss_dict[key], self._total_steps)
                self.writer.add_scalar('Lr', self._model._optimizer.param_groups[0]['lr'], self._total_steps)
                self._last_save_time = time.time()
            # if i_train_batch == 20:
            #     break

    def _display_terminal(self, iter_start_time, i_epoch, i_train_batch, num_batches):
        errors = self._model.get_current_errors()
        t = (time.time() - iter_start_time) 
        start_time = time.strftime("%H:%M", time.localtime(iter_start_time))
        output = "Time {}\tBatch Time {:.2f}\t Epoch [{}]([{}/{}])\t loss {:.4f}\t".format(
                                        start_time, t, 
                                        i_epoch, i_train_batch, num_batches,
                                        errors['loss'])
        for task in args.tasks:
            output += 'loss_{} {:.4f}\t'.format(task, errors['loss_{}'.format(task)])
        print(output)
    def _validate(self, i_epoch):
        val_start_time = time.time()
        # set model to eval
        self._model.set_eval()
        eval_per_task = {}
        tasks = copy(args.tasks)
        if 'FA' in tasks:
            tasks.remove('FA')
            eval_per_task['FA'] = []
        if 'VAD' in tasks:
            tasks.remove('VAD')
            eval_per_task['VAD'] = []
        hiddens = dict([(t, None) for t in args.tasks])
        video_name = None
        for task in tasks:
            track_val_preds = {'preds':[]}
            track_val_labels = {'labels':[]}
            val_errors = OrderedDict()
            data_loader = self.validation_dataloaders[task]
            for i_val_batch, val_batch in tqdm(enumerate(data_loader), total = len(data_loader)):
                # evaluate model
                wrapped_v_batch = {task: val_batch}
                if video_name is None:
                    video_name = val_batch['video'][0]
                else:
                    if video_name != val_batch['video'][0]:
                        hiddens = dict([(t, None) for t in args.tasks])
                self._model.set_input(wrapped_v_batch, input_tasks = [task])
                outputs, errors, hiddens = self._model.forward(return_estimates=True, 
                    input_tasks = [task], FA_teacher=FA_teacher, VAD_teacher=VAD_teacher, hiddens = hiddens)

                # store current batch errors
                for k, v in errors.items():
                    if k in val_errors:
                        val_errors[k] += v # accmulate over iters
                    else:
                        val_errors[k] = v
                #store the predictions and labels
                track_val_preds['preds'].append(outputs[task][task])
                track_val_labels['labels'].append(wrapped_v_batch[task]['label'])
                # if i_val_batch == 100:
                #     break
            # normalize errors
            for k in val_errors.keys():
                val_errors[k] /= len(data_loader)
            # calculate metric
            preds = np.concatenate(track_val_preds['preds'], axis=0)
            labels = np.concatenate(track_val_labels['labels'], axis=0)
            B, N = preds.shape[:2]
            metric_func = self._model.get_metrics_per_task()[task]
            eval_items, eval_res = metric_func(preds.reshape(B*N, -1).squeeze(), labels.reshape(B*N, -1).squeeze())
            now_time = time.strftime("%H:%M", time.localtime(val_start_time))
            output = "{} Validation {}: Epoch [{}] Step [{}] loss {:.4f} Eval_0 {:.4f} Eval_1 {:.4f}".format(task, 
                now_time, i_epoch, self._total_steps, val_errors['loss_{}'.format(task)], eval_items[0], eval_items[1])
            if 'FA' in args.tasks:
                output += " auxillary task FA loss: {:.4f}".format(val_errors['loss_FA'])
                eval_per_task['FA'].append(val_errors['loss_FA'])
            if 'VAD' in args.tasks:
                output += " auxillary task VAD loss: {:.4f}".format(val_errors['loss_VAD'])
                eval_per_task['VAD'].append(val_errors['loss_VAD'])
            print(output)
            eval_per_task[task] = [eval_items, eval_res]

        print("Validation Performance:")
        output = ""
        for task in tasks:
            output += '{} Metric: {:.4f}   '.format(task, eval_per_task[task][1])
        print(output)
        # set model back to train
        self._model.set_train()

        for task in args.tasks:
            save_dir = 'Val_{}'.format(task)
            if (task != 'FA') and (task != 'VAD'):
                if task == 'AU' or task == 'EXPR':
                    name0, name1 = 'F1', 'Acc'
                else:
                    name0, name1 = 'Valence_CCC', 'Arousal_CCC'
                self.writer.add_scalar(save_dir+'/'+name0, eval_per_task[task][0][0], i_epoch)
                self.writer.add_scalar(save_dir+'/'+name1, eval_per_task[task][0][1], i_epoch)
            else:
                self.writer.add_scalar(save_dir+'/'+'metric', np.mean(eval_per_task[task]), i_epoch)
                print('{} Metric: {:.4f}   '.format(task, np.mean(eval_per_task[task])))
                eval_per_task[task] = [0 , np.mean(eval_per_task[task])]
        return dict([(k, eval_per_task[k][1]) for k in args.tasks]) 

if __name__ == "__main__":
    trainer = Trainer()