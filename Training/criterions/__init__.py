# import six
# import sys
# import os
# from os.path import join as pjoin
# import numpy as np
# import random
# from PIL import Image
# import numbers
# import torchvision.transforms as transforms
# from torchvision.transforms import functional as TF
# import torch
# import torch.nn as nn
# from torch.autograd import Variable
# import torch.nn.functional as F
# from sklearn.metrics import f1_score
# from ..PATH import PATH


# class AU_Losses(object):
#     def __init__(self, AU_criterion):
#         self.criterion = AU_criterion
#         self.class_num = len(PATH().Aff_wild2.categories['AU'])
#     def get_task_loss(self):
#         if self.criterion == 'BCE':
#             task_loss = nn.BCEWithLogitsLoss().cuda()
#         return task_loss
#     def get_distillation_loss(self):
#         #distillation_loss use the cross entropy loss
#         def bce_with_logits(x, y):
#             y = torch.sigmoid(y)
#             return F.binary_cross_entropy_with_logits(x, y)
#         return bce_with_logits


# class EXPR_Losses(object):
#     def __init__(self, EXPR_criterion, temperature = 1):
#         self.class_num = len(PATH().Aff_wild2.categories['EXPR'])
#         self.criterion = EXPR_criterion
#         self.temperature = temperature
#     def get_task_loss(self):
#         if self.criterion == 'CE':
#             task_loss = nn.CrossEntropyLoss().cuda()
#         return task_loss
#     def get_distillation_loss(self):
#         def distill(y, teacher_pred, T=self.temperature):
#             return nn.KLDivLoss(reduction = 'batchmean')(F.log_softmax(y/T, dim=-1), F.softmax(teacher_pred/T, dim=-1))
#         return distill

# class VA_Losses(object):
#     def __init__(self, VA_criterion, uncertainty = True):
#         self.criterion = VA_criterion
#         self.uncertainty = uncertainty

#     def get_task_loss(self):
#     	if self.criterion == 'NNL':
#     		if not uncertainty:
#     			raise ValueError("Negative Log-likelihood is applied when uncertainty estimation is on.")
    		
#         ccc_loss = CCCLoss(self.digitize_num)
#         if self.digitize_num !=1:
#             if self.criterion == 'CCC_CE':
#                 classification_loss = Custom_CrossEntropyLoss(self.digitize_num)
#             elif self.criterion == 'CCC_FocalLoss':
#                 classification_loss = FocalLoss(self.digitize_num, self._opt.batch_size, activation = 'softmax')
#             def criterion_task(x, y):
#                 N = self.digitize_num 
#                 loss_v = (self._opt.lambda_ccc * ccc_loss(x[:, :N], y[:, :1]) + classification_loss(x[:, :N], y[:, :1]))
#                 loss_a = (self._opt.lambda_ccc * ccc_loss(x[:, N:], y[:, 1:]) + classification_loss(x[:, N:], y[:, 1:]))
#                 return loss_v, loss_a
#         else:
#             def criterion_task(x, y):
#                 N = self.digitize_num 
#                 return self._opt.lambda_V* ccc_loss(x[:, :N], y[:, :1]) + self._opt.lambda_A * ccc_loss(x[:, N:], y[:, 1:])
#         task_loss = criterion_task
#         return task_loss
#     def get_distillation_loss(self):
#         def distill(y, teacher_pred, T=self.temperature):
#             N = self.digitize_num 
#             loss_v = nn.KLDivLoss(reduction = 'batchmean')(F.log_softmax(y[:, :N]/T, dim=-1), F.softmax(teacher_pred[:, :N]/T, dim=-1))
#             loss_a = nn.KLDivLoss(reduction = 'batchmean')(F.log_softmax(y[:, N:]/T, dim=-1), F.softmax(teacher_pred[:, N:]/T, dim=-1))
#             return loss_v, loss_a
#         distillation_loss = distill
#         return distillation_loss
