"""
All supported optimizers, learning rate scheduler, loss functions (criterions) and evaluation metrics
"""
import torch
import time
import os
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional
from utils.validation import NLL, Accuracy, MSE, MAE, NLL_Regression, KLD_Regression
from utils.misc import rename
__all__ = ['Optimizer', 'Scheduler', 'Criterion', 'Metric']

class Optimizer:
    def __init__(self):
        pass
    @classmethod
    def get(
        self,
        model: nn.Module,
        optimizer: str,
        lr: float,
        wd: float = 0.,
        momentum: float = 0.,
        betas: List[float] = (0.9, 0.999)
        ): 

        if optimizer.lower() == 'sgd':
            optim = torch.optim.SGD(model.parameters(), 
                lr=lr, 
                momentum = momentum,
                weight_decay = wd,
                )
        elif optimizer.lower() == 'adam':
            optim = torch.optim.Adam(model.parameters(),
                lr=lr,
                weight_decay = wd,
                betas = betas
                )
        elif optimizer.lower() == 'none':
            optim = Optimizer()
        else:
            raise ValueError("Optimizer {} not supported".format(optimizer))
        return optim

    def step(self):
        pass

    def to(self):
        pass

from torch.optim.lr_scheduler import _LRScheduler
class Scheduler:
    def __init__(self):
        pass
    @classmethod
    def get(
        self,
        lr_scheduler: str,
        optimizer: torch.optim.Optimizer,
        step_size: int, # for StepLR only (epochs)
        T_max: int, # for CosineAnnealingLR only (iterations)
        gamma: float = 0.1, # for StepLR only (float number in 0 to 1),
        ):

        if lr_scheduler.lower() == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                step_size,
                gamma)
        elif lr_scheduler.lower() == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                T_max)
        elif lr_scheduler.lower() == 'none':
            scheduler = Scheduler()
        else:
            raise ValueError("lr_scheduler {} not supported".format(lr_scheduler))
        return scheduler
    
    def step(self):
        pass
    
    def to(self):
        pass

class Criterion:
    def __init__(self):
        pass
    @classmethod
    def get(
        self,
        loss: str,
        num_classes: int,
        reduction: str = 'mean',
        weight: Optional[torch.Tensor] = None,
        pos_weight: Optional[torch.Tensor] = None, # only for BCE loss
        ): 
        loss_name = loss
        if loss.lower() in ['ce', 'bce', 'mse', 'l1', 'l1_loss']:
            if loss.lower() == 'ce':
                loss = nn.CrossEntropyLoss(reduction = reduction, weight = weight)
            elif loss.lower() == 'bce':
                loss = nn.BCEWithLogitsLoss(reduction = reduction, 
                    weight = weight, 
                    pos_weight = pos_weight)
            elif loss.lower() == 'mse':
                loss = nn.MSELoss(reduction = reduction)
            elif loss.lower() == 'l1' or loss.lower() == 'l1_loss':
                loss = nn.L1Loss(reduction = reduction)
            # function wrapper to get the first num_classes from pred
            @rename(loss_name)
            def inner_func(pred, target):
                pred = pred[..., :num_classes]
                return loss(pred, target)
            return inner_func
        else:
            if loss.lower() == 'nll_reg' or loss.lower() == 'nll_regression':
                loss = NLL_Regression(num_classes = num_classes, reduction = reduction)
            elif loss.lower() == 'kld_reg' or loss.lower() == 'kld_regression':
                loss = KLD_Regression(num_classes = num_classes, reduction = reduction)
            else:
                raise ValueError("loss {} not supported".format(loss))
            return loss

class Metric:
    def __init__(self):
        pass 
    @classmethod
    def get(
        self,
        metric: str,
        num_classes: int,
        reduction: str = 'mean',
        apply_activation: str = 'softmax'
        ): 
        if metric.lower() == 'mse':
            m_func = MSE(num_classes=num_classes, reduction = reduction)
        elif metric.lower() == 'mae':
            m_func = MAE(num_classes=num_classes, reduction = reduction)
        elif metric.lower() == 'acc' or metric.lower() == 'accuracy':
            m_func = Accuracy(num_classes=num_classes, apply_activation=apply_activation)
        elif metric.lower() == 'nll':
            m_func = NLL(num_classes=num_classes, reduction = reduction, apply_activation=apply_activation)
        elif metric.lower() == 'nll_reg' or loss.lower() == 'nll_regression':
            m_func = NLL_Regression(num_classes=num_classes,reduction = reduction)
        else:
            raise ValueError("Metric {} not supported".format(metric))
        return m_func


