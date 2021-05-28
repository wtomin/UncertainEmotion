from PATH import PATH
import torch
import torch.nn as nn
from collections import OrderedDict
from torch.autograd import Variable
from criterions.optim import Scheduler, Criterion, Optimizer, Metric
from utils.validation import AU_metric, EXPR_metric, VA_metric, FA_metric, get_mean_sigma
import os
import numpy as np
import torch.nn.functional as F
from copy import deepcopy, copy

class ModelWrapper(object):
    def __init__(self, STModel, name, tasks, checkpoints_dir, 
        loggings_dir, load_epoch, batch_size, seq_len, image_size,
        lr, lr_policy, lr_decay_epochs, T_max, optimizer, wd, 
        gpu_ids,
        AU_criterion, EXPR_criterion, VA_criterion, FA_criterion,
        lambda_AU, lambda_EXPR, lambda_VA, lambda_FA,
        is_train = True,
        cuda = True,):

        self._name = name
        self._model = STModel
        self.tasks = tasks
        self.checkpoints_dir = checkpoints_dir
        self.loggings_dir = loggings_dir
        self.load_epoch = load_epoch
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.image_size = image_size
        self._is_train = is_train
        self.lr = lr
        self.lr_policy = lr_policy
        self.T_max = T_max
        self.lr_decay_epochs = lr_decay_epochs
        self.optimizer = optimizer
        self.wd = wd
        self.cuda = cuda
        self._gpu_ids = gpu_ids
        self._save_dir = os.path.join(self.checkpoints_dir, self.name)
        self._Tensor = torch.cuda.FloatTensor if self.cuda else torch.Tensor

        #
        categories = PATH().Aff_wild2.categories
        self._output_size_per_task = {'AU': len(categories['AU']), 'EXPR': len(categories['EXPR']), 
        'VA': len(categories['VA'])*20 ,
        'FA': 68*2}
        self.categories = categories

        self._criterions_per_task = {'AU': AU_criterion, 
        'EXPR': EXPR_criterion, 'VA': VA_criterion, 'FA': FA_criterion}
        self.lambda_AU = lambda_AU
        self.lambda_EXPR = lambda_EXPR
        self.lambda_VA = lambda_VA
        self.lambda_FA = lambda_FA

        # init train variables
        if self._is_train:
            self._model.train()
            self._init_train_vars()
        else:
            self._model.eval()

        # load networks and optimizers
        if load_epoch > 0:
            self.load()

        # prefetch variables
        self._init_prefetch_inputs()

        # init
        self._init_losses()
    @property
    def name(self):
        return self._name

    @property
    def is_train(self):
        return self._is_train
    @property
    def lambdas_per_task(self):
        lambda_dict =  {'AU': self.lambda_AU, 'EXPR': self.lambda_EXPR, 
        'VA': self.lambda_VA, 'FA': self.lambda_FA}
        return lambda_dict

    def update_lambda(self, no_improve_n_epochs):
        for key in no_improve_n_epochs.keys():
            n_epochs = no_improve_n_epochs[key]
            assert isinstance(n_epochs, int), "number of epochs should be an integer"
            if n_epochs > 1:
                setattr(self, 'lambda_{}'.format(key), np.log2(n_epochs))
    def normalize_lambda(self, lambda_dict):
        summation = sum([lambda_dict[key] for key in lambda_dict.keys()])
        return dict([(k, lambda_dict[k]/summation) for k in lambda_dict.keys()])
    
    def load(self):
        load_epoch = self.load_epoch
        # load feature extractor
        self._load_network(self._model, self.name, load_epoch)
        self._load_optimizer(self._optimizer, 'OPT', load_epoch)

    def save(self, label):
        """
        save network, the filename is specified with the sofar tasks and iteration
        """
        self._save_network(self._model, self.name, label)
        # save optimizers
        self._save_optimizer(self._optimizer, 'OPT', label)

    def _init_train_vars(self):
        self._optimizer = Optimizer().get(self._model, self.optimizer,
            lr=self.lr, wd=self.wd)
        self._LR_scheduler = Scheduler().get(self.lr_policy,self._optimizer, 
            step_size = self.lr_decay_epochs, T_max = self.T_max)

    def _format_label_tensor(self, task):
        _Tensor_Long = torch.cuda.LongTensor if self.cuda else torch.LongTensor
        _Tensor_Float = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor
        if task =='AU' or task == 'VA' or task =='FA':
            return _Tensor_Float(self.batch_size, self.seq_len, self._output_size_per_task[task])
        elif task == 'EXPR':
            return _Tensor_Long(self.batch_size, self.seq_len)

    def _init_prefetch_inputs(self):
        if 'FA' in self.tasks:
            input_tasks = copy(self.tasks)
            input_tasks.remove('FA')
        else:
            input_tasks = self.tasks
        self._input_image = OrderedDict([(task, self._Tensor(self.batch_size, self.seq_len, 3, self.image_size, self.image_size)) for task in input_tasks])  
        self._label = OrderedDict([(task, self._format_label_tensor(task)) for task in input_tasks])
    def _init_losses(self):
        # get the training loss
        criterions = {}
        criterions['AU'] = Criterion().get(self._criterions_per_task['AU'], len(self.categories['AU']))
        criterions['EXPR'] = Criterion().get(self._criterions_per_task['EXPR'], len(self.categories['EXPR']))
        criterions['VA'] = Criterion().get(self._criterions_per_task['VA'], len(self.categories['VA']) * 20)
        criterions['FA'] = Criterion().get(self._criterions_per_task['FA'], 68*2)
        self._criterions_per_task = criterions

        # metrics = {}
        # metrics['AU'] = Metric().get(self._metrics_per_task['AU'], len(self.categories['AU']))
        # metrics['EXPR'] = Metric().get(self._metrics_per_task['EXPR'], len(self.categories['EXPR']))
        # metrics['VA'] = Metric().get(self._metrics_per_task['VA'], len(self.categories['VA']))
        # metrics['FA'] = Metric().get(self._metrics_per_task['FA'], 68*2)
        # self._metrics_per_task = metrics

    def set_input(self, input, input_tasks = None):
        """
        During training, the input will be only related to the current task, because the input only has one type of label
        During validation, because the current model needs to be evaluated on all sofar tasks, the task needs to be specified
        """
        tasks = copy(self.tasks) if input_tasks is None else input_tasks
        if 'FA' in tasks:
            tasks.remove('FA') # face alignment is an auxillary task, which does not have training labels
        for t in tasks:
            self._input_image[t].resize_(input[t]['image'].size()).copy_(input[t]['image'])
            self._label[t].resize_(input[t]['label'].size()).copy_(input[t]['label'])
            if len(self._gpu_ids) > 0:
                self._input_image[t] = self._input_image[t].cuda()
                self._label[t] = self._label[t].cuda()
    def set_train(self):
        self._model.train()
        self._is_train = True

    def set_eval(self):
        self._model.eval()
        self._is_train = False

    def forward(self, return_estimates=False, input_tasks = None,
        FA_teacher = None, hiddens = None):
        # validation the eval_task
        val_dict = dict() 
        out_dict = dict()
        loss = 0.
        if not self._is_train:
            tasks = copy(self.tasks) if input_tasks is None else input_tasks
            if 'FA' in tasks:
                tasks.remove('FA')
            for t in tasks:
                with torch.no_grad():
                    input_image = Variable(self._input_image[t])
                    label = Variable(self._label[t])
                    output, hiddens = self._model(input_image, hiddens)
                criterion_task = self._criterions_per_task[t]
                B, N, C  = output[t].size()
                loss_task = criterion_task(output[t].view(B*N, C), label.view(B*N, -1).squeeze(-1)) 
                val_dict['loss_'+t] = loss_task.item()
                loss += self.normalize_lambda(self.lambdas_per_task)[t] * loss_task
                if FA_teacher is not None and 'FA' in self.tasks:
                    with torch.no_grad():
                        FA_label = FA_teacher(input_image.view((B*N, ) + input_image.size()[2:]))
                        B, N, C  = output['FA'].size()
                        loss_FA = self._criterions_per_task['FA'](output['FA'].view(B*N, C), FA_label.view(B*N, -1).squeeze(-1))
                    if 'loss_FA' not in val_dict.keys():
                        val_dict['loss_FA'] = []
                    val_dict['loss_FA'].append(loss_FA.item() * (1/len(tasks)))
                    loss += self.normalize_lambda(self.lambdas_per_task)['FA'] * (1/len(tasks)) * loss_FA

                if return_estimates:
                    for task in self.tasks:
                        out_dict[t] = self._format_estimates(output)
                else:
                    for task in self.tasks:
                        out_dict[t] = dict([(key, output[key].cpu().numpy()) for key in output.keys()])
            val_dict['loss'] = loss.item()
        else:
            raise ValueError("Do not call forward function in training mode. USE optimize_parameters() INSTEAD.")
        if 'loss_FA' in val_dict:
            val_dict['loss_FA'] = sum(val_dict['loss_FA'])
        return out_dict, val_dict, hiddens
    def _format_estimates(self, output):
        estimates = {}
        for task in output.keys():
            if task == 'AU':
                o = (torch.sigmoid(output['AU'].cpu())>0.5).type(torch.LongTensor)
                estimates['AU'] = o.numpy()
            elif task == 'EXPR':
                o = F.softmax(output['EXPR'].cpu(), dim=-1).argmax(-1).type(torch.LongTensor)
                estimates['EXPR'] = o.numpy()
            elif task == 'VA':
                N = 20
                v = F.softmax(output['VA'][:,:, :N].cpu(), dim=-1).numpy()
                a = F.softmax(output['VA'][:,:, N:].cpu(), dim=-1).numpy()
                bins = np.linspace(-1, 1, num=N)
                v = (bins * v).sum(-1)
                a = (bins * a).sum(-1)
                estimates['VA'] = np.stack([v, a], axis = -1)
            elif task == 'FA':
                estimates['FA'] = output['FA'].cpu().numpy()
        return estimates
    def optimize_parameters(self, FA_teacher = None):
        train_dict = dict()
        loss = 0.
        if self._is_train:
            tasks = copy(self.tasks)
            if 'FA' in self.tasks:
                tasks.remove("FA")
            for t in tasks:
                input_image = Variable(self._input_image[t])
                label = Variable(self._label[t])
                output, _ = self._model(input_image)
                criterion_task = self._criterions_per_task[t]
                B, N, C  = output[t].size()
                loss_task = criterion_task(output[t].view(B*N, C), label.view(B*N, -1).squeeze(-1)) 
                train_dict['loss_'+t] = loss_task.item()
                loss += self.normalize_lambda(self.lambdas_per_task)[t] * loss_task
                if FA_teacher is not None and 'FA' in self.tasks:
                    with torch.no_grad():
                        FA_label = FA_teacher(input_image.view((B*N,) + input_image.size()[2:]))
                    B, N, C  = output['FA'].size()
                    loss_FA = self._criterions_per_task['FA'](output['FA'].view(B*N, C), FA_label.view(B*N, -1).squeeze(-1))
                    if 'loss_FA' not in train_dict.keys():
                        train_dict['loss_FA'] = []
                    train_dict['loss_FA'].append(loss_FA.item() * (1/len(tasks)))
                    loss += self.normalize_lambda(self.lambdas_per_task)['FA'] * (1/len(tasks)) * loss_FA
            train_dict['loss'] = loss.item()
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()
            if 'loss_FA' in train_dict.keys():
                train_dict['loss_FA'] = sum(train_dict['loss_FA'])
            self.loss_dict = train_dict
        else:
            raise ValueError("Do not call optimize_parameters function in test mode. USE forward() INSTEAD.")
    
    # def optimize_parameters_kd(self, teacher_model): # knowledge distillation
    #     train_dict = dict()
    #     loss = 0.
    #     loss_per_task = {'EXPR': 0, 'valence':0, 'arousal':0, 'AU':0}
    #     if self._is_train:
    #         for t in self._opt.tasks:
    #             input_image = Variable(self._input_image[t])
    #             output = self.resnet50_GRU(input_image)
    #             label = Variable(self._label[t])
    #             with torch.no_grad():
    #                 teacher_preds = teacher_model.resnet50_GRU(input_image)
    #             for task in self._opt.tasks:
    #                 distillation_task = self._criterions_per_task[task].get_distillation_loss()
    #                 B, N, C  = output['output'][task].size()
    #                 loss_task = distillation_task(output['output'][task].view(B*N, C), teacher_preds['output'][task].view(B*N, C))
    #                 if task == t:
    #                     if task!= 'VA':
    #                         criterion_task = self._criterions_per_task[t].get_task_loss()
    #                         B, N, C  = output['output'][t].size()
    #                         loss_task = self._opt.lambda_teacher * loss_task + (1 - self._opt.lambda_teacher) * criterion_task(output['output'][t].view(B*N, C), label.view(B*N, -1).squeeze())
                        
    #                     else:
    #                         criterion_task = self._criterions_per_task[t].get_task_loss()
    #                         loss_v, loss_a = criterion_task(output['output'][t].view(B*N, C), label.view(B*N, -1).squeeze())
    #                         loss_task = [self._opt.lambda_teacher * loss_task[0] + (1 - self._opt.lambda_teacher) *loss_v,
    #                                     self._opt.lambda_teacher * loss_task[1] + (1 - self._opt.lambda_teacher) *loss_a,] 
    #                 if task!= 'VA':
    #                     loss_per_task[task] += loss_task.item()
    #                     loss += loss_task
    #                 else:
    #                     loss_v, loss_a = loss_task
    #                     loss_per_task['valence'] += loss_v.item()
    #                     loss_per_task['arousal'] += loss_a.item()
    #                     loss += loss_v + loss_a
    #         loss = loss/len(self._opt.tasks)
    #         for key in loss_per_task.keys():
    #             loss_task = loss_per_task[key]
    #             train_dict['loss_'+key] = loss_task/len(self._opt.tasks)
    #         train_dict['loss'] = loss.item()
    #         self._optimizer.zero_grad()
    #         loss.backward()
    #         self._optimizer.step()
    #         self.loss_dict = train_dict
    #     else:
    #         raise ValueError("Do not call optimize_parameters function in test mode. USE forward() INSTEAD.")
    def get_current_errors(self):
        return self.loss_dict
    def get_metrics_per_task(self):
        return {"AU": AU_metric, "EXPR": EXPR_metric, "VA": VA_metric, "FA": FA_metric}
    def get_current_LR(self):
        LR = []
        for param_group in self._optimizer.param_groups:
            LR.append(param_group['lr'])
        print('current learning rate: {}'.format(np.unique(LR)))

    def _save_optimizer(self, optimizer, optimizer_label, epoch_label):
        save_filename = 'opt_epoch_%s_%s.pth' % (epoch_label, optimizer_label)
        save_path = os.path.join(self._save_dir, save_filename)
        torch.save(optimizer.state_dict(), save_path)

    def _load_optimizer(self, optimizer, optimizer_label, epoch_label):
        load_filename = 'opt_epoch_%s_%s.pth' % (epoch_label, optimizer_label)
        load_path = os.path.join(self._save_dir, load_filename)
        assert os.path.exists(
            load_path), 'Weights file not found. Have you trained a model!? We are not providing one' % load_path

        optimizer.load_state_dict(torch.load(load_path))
        print ('loaded optimizer: %s' % load_path)

    def _save_network(self, network, network_label, epoch_label):
        save_filename = 'net_epoch_%s_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self._save_dir, save_filename)
        save_dict = {'state_dict': network.state_dict(), 'epoch': epoch_label}
        torch.save(save_dict, save_path)
        print ('saved net: %s' % save_path)

    def _load_network(self, network, network_label, epoch_label):
        load_filename = 'net_epoch_%s_%s.pth' % (epoch_label, network_label)
        load_path = os.path.join(self._save_dir, load_filename)
        assert os.path.exists(
            load_path), 'Weights file not found. Have you trained a model!? We are not providing one' % load_path
        from ..utils import map_location
        checkpoint = torch.load(load_path, map_location = map_location(self.cuda))
        network.load_state_dict(checkpoint['state_dict'])
        print ('loaded net: %s' % load_path)

    def print_network(self, network):
        num_params = 0
        for param in network.parameters():
            num_params += param.numel()
        print(network)
        print('Total number of parameters: %d' % num_params)

    # def _get_scheduler(self, optimizer, lr_policy,
    #     lr_decay_epochs):
    #     if lr_policy == 'step':
    #         scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_decay_epochs, gamma=0.1)
    #     elif lr_policy == 'plateau':
    #         scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, threshold=0.01, patience=2)
    #     else:
    #         return NotImplementedError('learning rate policy [%s] is not implemented', lr_policy)
    #     return scheduler