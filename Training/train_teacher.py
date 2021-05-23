import time
import argparse
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
#########################################################################
parser = argparse.ArgumentParser()
######### Losses #############
parser.add_argument('--AU_criterion', type=str, default = 'bce')
parser.add_argument('--EXPR_criterion', type=str, default = 'ce')
parser.add_argument('--VA_criterion', type=str, default = 'cce+ccc')
parser.add_argument('--FA_criterion', type=str, default= 'l1_loss')
parser.add_argument('--lambda_AU', type=float, default=1)
parser.add_argument('--lambda_EXPR', type=float, default=1)
parser.add_argument('--lambda_VA', type=float, default=1)
parser.add_argument('--lambda_FA', type=float, default=1)
########## Data and tasks #########
parser.add_argument('--dataset_names', type=str, default = ['Mixed_EXPR','Mixed_AU','Mixed_VA'],nargs="+")
parser.add_argument('--tasks', type=str, default = ['EXPR','AU','VA'],nargs="+")
parser.add_argument('--seq_len', type=int, default= 30, help='length of input seq ')
parser.add_argument('--fps', type=int, default=30, help=
    "Changing the fps to some integer smaller than 30 can change the sampling rate")
parser.add_argument('--batch_size', type=int, default= 2, help='input batch size per task')
parser.add_argument('--image_size', type=int, default= 112, help='input image size') 
#parser.add_argument('--uncertainty', action='store_true', help='whether to predict uncertainty for emotion tasks.')

########### Ablation study: w/o auxillary task; Transformer or RNN #########
parser.add_argument('--TModel', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU, Transformer)')
parser.add_argument('--auxillary', action='store_true', help=
    "Whether to train face alignment as an auxillary task.")
########## temporal model definition #########
parser.add_argument('--nhead', type=int, default=2,
                    help='the number of heads in the encoder/decoder of the transformer model')
parser.add_argument('--nhid', type=int, default=128,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers in the temporal model')
########## Training setup ############
parser.add_argument('--load_epoch', type=int, default=-1, 
    help='which epoch to load? set to -1 to use latest cached model')
parser.add_argument('--lr', type=float, default=1e-3, 
    help= "The initial learning rate")
parser.add_argument('--lr_policy', type=str, default='step', choices=['step', 'cosine'])
parser.add_argument('--lr_decay_epochs', type=int, default=10, help='reduce the lr to 0.1*lr for every # epochs')
parser.add_argument('--T_max', type=int, default=10000, help='the period for the cosine annealing (# iterations)')
parser.add_argument('--weight_decay', type=float, default=0., help='weight decay')
parser.add_argument('--nepochs', type=int, default=36)
parser.add_argument('--optimizer', type=str, default='Adam')
parser.add_argument('--gpu_ids', type=str, default='0', nargs='+',
    help='gpu ids: e.g. 0 , 0 1 2. use -1 for CPU')
parser.add_argument('--cuda', action='store_true', help="Whether to use GPU")
parser.add_argument('--print_freq_s', type=int, default= 5, help='print the training loss after every # seconds')
parser.add_argument('--save_freq_s', type=int, default= 10,
    help= 'save the training losses to the summary writer every # seconds.')
parser.add_argument('--n_threads_train', default=8, type=int, help='# threads for loading data')
parser.add_argument('--n_threads_test', default=8, type=int, help='# threads for loading data')
parser.add_argument('--name', type=str, default='experiment_1', help='name of the experiment. It decides where to store samples and models')
parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
parser.add_argument('--loggings_dir', type=str, default='./loggings', help='loggings are saved here')

args = parser.parse_args()
prepare_arguments(args, is_train=True)

if args.auxillary:
    from utils.misc import mobile_facenet
    print("Training model with an auxillary task: face alignment.")
    args.tasks = args.tasks + ['FA']
    FA_teacher = mobile_facenet(pretrained=True, cuda=args.cuda)
else:
    FA_teacher = None
FA_teacher.eval()
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
            tasks = args.tasks, batch_size = args.batch_size, seq_len = args.seq_len, fps = args.fps,
            transform = train_transforms(args.image_size))
        self.training_dataloaders = self.training_dataloaders.load_multitask_train_data()
        self.validation_dataloaders = Multitask_DatasetDataLoader(
            train_mode = 'Validation', num_threads = args.n_threads_test, dataset_names=args.dataset_names,
            tasks = args.tasks, batch_size = args.batch_size, seq_len = args.seq_len, fps = 30, # validation set always sample by 30 fps
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
        self._current_val_acc = 0.

        for i_epoch in range(args.load_epoch + 1, args.nepochs + 1):
            epoch_start_time = time.time()
            self._model.get_current_LR()
            # train epoch
            self._train_epoch(i_epoch)
            self.training_dataloaders.reset()
            if args.lr_policy == 'step':
                self._model._LR_scheduler.step()
            val_acc = self._validate(i_epoch)
            if val_acc > self._current_val_acc:
                print("validation acc improved, from {:.4f} to {:.4f}".format(self._current_val_acc, val_acc))
                print('saving the model at the end of epoch %d, steps %d' % (i_epoch, self._total_steps))
                #self._model.save(0) # only save the best on validation set
                self._current_val_acc = val_acc
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

            self._model.optimize_parameters(FA_teacher=FA_teacher)

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
            if i_train_batch == 100:
                break

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
        for task in tasks:
            track_val_preds = {'preds':[]}
            track_val_labels = {'labels':[]}
            val_errors = OrderedDict()
            data_loader = self.validation_dataloaders[task]
            for i_val_batch, val_batch in tqdm(enumerate(data_loader), total = len(data_loader)):
                # evaluate model
                wrapped_v_batch = {task: val_batch}
                self._model.set_input(wrapped_v_batch, input_tasks = [task])
                outputs, errors = self._model.forward(return_estimates=True, 
                    input_tasks = [task], FA_teacher=FA_teacher)

                # store current batch errors
                for k, v in errors.items():
                    if k in val_errors:
                        val_errors[k] += v # accmulate over iters
                    else:
                        val_errors[k] = v
                #store the predictions and labels
                track_val_preds['preds'].append(outputs[task][task])
                track_val_labels['labels'].append(wrapped_v_batch[task]['label'])
                if i_val_batch == 100:
                    break
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
            print(output)
            eval_per_task[task] = [eval_items, eval_res]

        print("Validation Performance:")
        output = ""
        for task in eval_per_task.keys():
            output += '{} Metric: {:.4f}   '.format(task, eval_per_task[task][1])
        print(output)
        # set model back to train
        self._model.set_train()

        for task in args.tasks:
            save_dir = 'Val_{}'.format(task)
            if task != 'FA':
                if task == 'AU' or task == 'EXPR':
                    name0, name1 = 'F1', 'Acc'
                else:
                    name0, name1 = 'Valence_CCC', 'Arousal_CCC'
                self.writer.add_scalar(save_dir+'/'+name0, eval_per_task[task][0][0], i_epoch)
                self.writer.add_scalar(save_dir+'/'+name1, eval_per_task[task][0][1], i_epoch)
            else:
                self.writer.add_scalar(save_dir+'/'+'metric', np.mean(eval_per_task[task]), i_epoch)
        return sum([eval_per_task[k][1] for k in tasks]) # only consider the tasks except for the auxillary task

if __name__ == "__main__":
    trainer = Trainer()