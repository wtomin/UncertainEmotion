import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from sklearn.metrics import f1_score
from typing import Type, Any, Callable, Union, List, Optional
EPS =  1e-8
def averaged_f1_score(input, target):
    N, label_size = input.shape
    f1s = []
    for i in range(label_size):
        f1 = f1_score(input[:, i], target[:, i])
        f1s.append(f1)
    return np.mean(f1s), f1s
def accuracy(input, target):
    assert len(input.shape) == 1
    return sum(input==target)/input.shape[0]
def averaged_accuracy(x, y):
    assert len(x.shape) == 2
    N, C =x.shape
    accs = []
    for i in range(C):
        acc = accuracy(x[:, i], y[:, i])
        accs.append(acc)
    return np.mean(accs), accs
def CCC_score(x, y):
    vx = x - np.mean(x)
    vy = y - np.mean(y)
    rho = np.sum(vx * vy) / (np.sqrt(np.sum(vx**2)) * np.sqrt(np.sum(vy**2)))
    x_m = np.mean(x)
    y_m = np.mean(y)
    x_s = np.std(x)
    y_s = np.std(y)
    ccc = 2*rho*x_s*y_s/(x_s**2 + y_s**2 + (x_m - y_m)**2)
    return ccc
class CCCLoss(nn.Module):
    def __init__(self ):
        super(CCCLoss, self).__init__() 
    def forward(self, x, y): 
        # the target y is continuous value (BS, )
        # the input x is either continuous value (BS, ) 
        y = y.view(-1)
        x = x.view(-1)
        vx = x - torch.mean(x) 
        vy = y - torch.mean(y) 
        rho =  torch.sum(vx * vy) / (torch.sqrt(torch.sum(torch.pow(vx, 2))) * torch.sqrt(torch.sum(torch.pow(vy, 2))) + EPS)
        x_m = torch.mean(x)
        y_m = torch.mean(y)
        x_s = torch.std(x)
        y_s = torch.std(y)
        ccc = 2*rho*x_s*y_s/(torch.pow(x_s, 2) + torch.pow(y_s, 2) + torch.pow(x_m - y_m, 2) + EPS)
        return 1-ccc
def VA_metric(x, y):
    items = [CCC_score(x[:,0], y[:,0]), CCC_score(x[:,1], y[:,1])]
    return items, sum(items)
def EXPR_metric(x, y): 
    if not len(x.shape) == 1:
        if x.shape[1] == 1:
            x = x.reshape(-1)
        else:
            x = np.argmax(x, axis=-1)
    if not len(y.shape) == 1:
        if y.shape[1] == 1:
            y = y.reshape(-1)
        else:
            y = np.argmax(y, axis=-1)

    f1 = f1_score(x, y, average= 'macro')
    acc = accuracy(x, y)
    return [f1, acc], 0.67*f1 + 0.33*acc
def AU_metric(x, y):
    f1_av,_  = averaged_f1_score(x, y)
    x = x.reshape(-1)
    y = y.reshape(-1)
    acc_av  = accuracy(x, y)
    return [f1_av, acc_av], 0.5*f1_av + 0.5*acc_av
def FA_metric(x, y):
    # L1 Loss
    assert x.shape == y.shape, 'The prediction and label must have the same shape'
    return [None, None], np.abs(x - y).mean()

def get_mean_sigma(pred):
    C_double = pred.size(-1)
    num_classes = C_double//2
    y_hat, sigma_square = pred[..., :num_classes], pred[..., num_classes:]
    sigma_square_pos = torch.log(1 + torch.exp(sigma_square)) + 1e-06
    return y_hat, sigma_square_pos
class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        pass
    def forward(self,):
        raise NotImplementedError
    def to(self, device):
        pass #default: nothing to put to device
    def load_state_dict(self, state_dict):
        pass # default: nothing to load
class NLL_Regression(CustomLoss):
    def __init__(
        self,
        num_classes: int,
        reduction: str = 'mean',
        apply_activation: str = 'none'
        )->None:
        super(NLL_Regression, self).__init__()
        self.num_classes = num_classes # the number of classes, not including variances outputs
        self.reduction = reduction
        self.optimal = 'min' # lower NLL is better
        self.apply_activation = apply_activation
    def forward(self, pred, target)-> float: 
        """The Negative Log-likelihood loss function (or metric function) for regression task only,
           The pred tensor can be separated into y_hat and log_sigma_square.

        Args:
            pred (torch.Tensor): shape (N, C+C), the last C dimensions correspond to the log_sigma**2. The first C dimensions correspond to the prediction vector
            target (torch.Tensor): shape (N, C)
        """
        N, C_double = pred.size()
        assert C_double%2==0, "When using NLL as regression loss, prediction vector dimension must be divisble by two."
        num_classes = C_double//2
        assert self.num_classes == num_classes, "Expect to get ({}) tensor, got ({}) tensor".format((N, self.num_classes), pred.size())
        y_hat, sigma_square = get_mean_sigma(pred)
        MSE = 0.5* ((target - y_hat)**2)
        #loss = (torch.exp(-log_sigma_square)*MSE + 0.5*log_sigma_square).mean(-1)# average over the num_classes
        loss = 0.5*torch.log(sigma_square) + torch.div(MSE, sigma_square) + 1e-6
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            raise ValueError("{} reduction not supported.".format(self.reduction))
class KLD_Regression(CustomLoss):
    def __init__(
        self,
        num_classes: int,
        reduction: str = 'mean',
        apply_activation: str = 'none'
        )->None:
        super(KLD_Regression, self).__init__()
        self.num_classes = num_classes # the number of classes, not including variances outputs
        self.reduction = reduction
        self.optimal = 'min' # lower NLL is better
        self.apply_activation = apply_activation
    def forward(self, pred, target)-> float: 
        """The Negative Log-likelihood loss function (or metric function) for regression task only,
           The pred tensor can be separated into y_hat and log_sigma_square.

        Args:
            pred (torch.Tensor): shape (N, C+C), the last C dimensions correspond to the log_sigma**2. The first C dimensions correspond to the prediction vector
            target (torch.Tensor): shape (N, C+C), the teacher soft labels
        """
        N, C_double = pred.size()
        assert C_double%2==0, "When using NLL as regression loss, prediction vector dimension must be divisble by two."
        num_classes = C_double//2
        assert self.num_classes == num_classes, "Expect to get ({}) tensor, got ({}) tensor".format((N, self.num_classes), pred.size())
        y_hat, sigma_square_hat = get_mean_sigma(pred)
        y, sigma_square = get_mean_sigma(target)
        loss = torch.log(torch.sqrt(sigma_square)) - torch.log(torch.sqrt(sigma_square_hat))
        loss += torch.div(sigma_square_hat + (y_hat - y)**2, 2*sigma_square+1e-6)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            raise ValueError("{} reduction not supported.".format(self.reduction))

def nll_regression(mean, sigma, target):
    loss =  0.5*torch.log(sigma) + 0.5* torch.div((mean - target)**2, sigma) + 1e-6
    return loss.mean()
class NLL(CustomLoss):
    def __init__(
        self,
        num_classes: int,
        reduction: str = 'mean',
        apply_activation : str = 'softmax'
        )-> None:
        super(NLL, self).__init__()
        self.num_classes = num_classes
        self.reduction = reduction
        self.optimal = 'min' # lower NLL is better
        self.apply_activation = apply_activation

    def forward(self, pred, target)-> float: 
        # pred: (N, C) tensor
        # target: (N, ) tensor
        pred = pred[...,:self.num_classes]
        if self.apply_activation == 'softmax':
            pred = F.softmax(pred, dim=-1)
        elif self.apply_activation == 'sigmoid':
            pred = nn.sigmoid(pred)

        N = pred.size(0) # the number of instances
        NLLLoss = nn.NLLLoss(reduction=self.reduction)
        NLL = NLLLoss(torch.log(pred+EPS), target)
        return NLL

class Accuracy(CustomLoss):
    def __init__(
        self,
        num_classes: int,
        reduction: str = 'mean', # not applied for accuracy
        apply_activation: str = 'softmax'
        )-> None: 
        super(Accuracy, self).__init__()
        self.num_classes = num_classes
        self.reduction = reduction
        self.optimal = 'max' # higher is better
        self.apply_activation = apply_activation

    def forward(self, pred, target)-> float: 
        # pred: (N, C) tensor
        # target: (N, ) tensor
        pred = pred[..., :self.num_classes]
        if self.apply_activation == 'softmax':
            pred = F.softmax(pred, dim=-1)
        elif self.apply_activation == 'sigmoid':
            pred = nn.sigmoid(pred)
        elif self.apply_activation == 'none':
            pass
        else:
            raise ValueError(f"Not support activation {self.apply_activation}")
        assert len(pred.size()) > 1, "Expected pred tensor to have dims > 2, got size :{}".format(pred.size())
        pred = torch.max(pred, dim=-1)[1]
        N = pred.size(0) # the number of instances
        assert pred.shape==target.shape
        return (pred.eq(target).sum()/float(N))

class MSE(CustomLoss):
    def __init__(
        self,
        num_classes: str,
        reduction: str = 'mean', 
        apply_activation: str = 'none', # not applied for MSE
        )-> None: 
        super(MSE, self).__init__()
        self.num_classes = num_classes
        self.reduction = reduction
        self.optimal = 'min' # lower is better
        self.apply_activation = apply_activation

    def forward(self, pred, target):
        pred = pred[..., :self.num_classes] 
        assert pred.size() == target.size(), "pred tensor must have the same size as the target tensor"
        return F.mse_loss(pred, target, reduction=self.reduction)

class MAE(CustomLoss):
    def __init__(
        self,
        num_classes: str,
        reduction: str = 'mean', 
        apply_activation: str = 'none', # not applied for MSE
        )-> None: 
        super(MAE, self).__init__()
        self.num_classes = num_classes
        self.reduction = reduction
        self.optimal = 'min' # lower is better
        self.apply_activation = apply_activation

    def forward(self, pred, target):
        pred = pred[..., :self.num_classes] 
        assert pred.size() == target.size(), "pred tensor must have the same size as the target tensor"
        return F.l1_loss(pred, target, reduction=self.reduction)
# def merge_all_metrics(
#     metrics_names: List,
#     metrics_values: Union[dict, List],
#     metrics_funcs: Union[dict, List],
#     metrics_weights: Union[dict, List]
#     )-> float: 
#     total_metric = 0
#     def get(input, id, name):
#         if isinstance(input, List):
#             return input[id]
#         elif isinstance(input, dict):
#             return input[name]
#         else:
#             raise ValueError("Expected input argument is list or dict types, get {}".format(input))
#     for i_metric, metric_name in enumerate(metrics_names):
#         metric_func = get(metrics_funcs, i_metric, metric_name)
#         metric_weight = get(metrics_weights, i_metric, metric_name)
#         optimal = metric_func.optimal # 'min' or 'max'
#         lambda_o = -1 if optimal == 'max' else 1
#         total_metric += lambda_o* get(metrics_values, i_metric, metric_name)* metric_weight
        
#     return total_metric



############ to device ##############
def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)

def scheduler_to(sched, device):
    for param in sched.__dict__.values():
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
